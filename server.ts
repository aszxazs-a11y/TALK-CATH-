import express from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const app = express();
app.use(express.json({ limit: '25mb' }));

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// ------------------------------------------------------------------
// Error helpers
// ------------------------------------------------------------------
function getErrorCode(err: any): string {
  const raw = String(err?.message || err || '');
  if (/TIMEOUT|timeout|deadline/i.test(raw)) return 'TIMEOUT';
  if (/429|RESOURCE_EXHAUSTED|RATE_LIMIT|quota/i.test(raw)) return 'RATE_LIMIT';
  if (/API_KEY_INVALID|API key not valid|invalid api key|PERMISSION_DENIED/i.test(raw)) return 'API_KEY_INVALID';
  if (/PARSE_FAIL|SyntaxError|Unexpected token/i.test(raw)) return 'PARSE_FAIL';
  if (/404|NOT_FOUND|is not found|not supported/i.test(raw)) return 'MODEL_NOT_FOUND';
  if (/503|UNAVAILABLE|high demand|overloaded/i.test(raw)) return 'MODEL_BUSY';
  if (/INVALID_ARGUMENT|400/i.test(raw)) return 'INVALID_REQUEST';
  return 'MODEL_ERROR';
}

function friendlyMessage(code: string): string {
  switch (code) {
    case 'TIMEOUT':
      return 'AI 응답이 늦어지고 있어요. 잠시 후 다시 시도해 주세요.';
    case 'RATE_LIMIT':
      return '잠깐 사이에 요청이 많았어요. 1분 정도 뒤에 다시 시도해 주세요.';
    case 'API_KEY_INVALID':
      return 'Gemini API 키를 확인해 주세요. (AI Studio의 Secrets 설정 또는 우측 상단 열쇠 버튼)';
    case 'MODEL_BUSY':
      return '지금 AI 서버가 붐벼요. 잠시 후 다시 시도해 주세요.';
    case 'INVALID_REQUEST':
      return '이미지를 읽지 못했어요. 다른 캡처(PNG, JPG)로 다시 시도해 주세요.';
    case 'PARSE_FAIL':
      return 'AI 답변을 정리하지 못했어요. 다시 시도해 주세요.';
    default:
      return '분석 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.';
  }
}

// ------------------------------------------------------------------
// JSON cleanup + safe defaults so the UI never breaks
// ------------------------------------------------------------------
const str = (v: any, fallback = '') => (typeof v === 'string' && v.trim() ? v.trim() : fallback);
const SPEEDS = ['fast', 'moderate', 'slow', 'late_night', 'mixed', 'unknown'];

function cleanAndSanitizeJson(rawText: string): any {
  let clean = (rawText || '').trim();
  if (clean.includes('```')) clean = clean.replace(/```(?:json)?/gi, '').trim();
  const first = clean.indexOf('{');
  const last = clean.lastIndexOf('}');
  if (first !== -1 && last > first) clean = clean.substring(first, last + 1);

  let data: any;
  try {
    data = JSON.parse(clean);
  } catch {
    throw new Error('PARSE_FAIL: AI 답변의 JSON 구조를 해석하지 못했습니다.');
  }

  data.is_valid_chat = data.is_valid_chat !== false;
  data.invalid_reason = str(data.invalid_reason);
  data.chat_type = data.chat_type === 'everyday_routine' ? 'everyday_routine' : 'meaningful';
  data.confidence = ['low', 'medium', 'high'].includes(data.confidence) ? data.confidence : 'medium';

  const t = Number(data.temperature);
  data.temperature = Number.isFinite(t) ? Math.max(0, Math.min(100, Math.round(t))) : 50;
  // A low score means there is a real signal (complaint, distance...), so it is never 'everyday'
  if (data.chat_type === 'everyday_routine' && data.temperature < 50) data.chat_type = 'meaningful';

  data.status_tags = (Array.isArray(data.status_tags) ? data.status_tags : [])
    .map((x: any) => String(x || '').replace(/^[#\s]+/, '').trim())
    .filter(Boolean)
    .slice(0, 4);

  data.summary = str(data.summary, '대화의 흐름을 바탕으로 정리한 결과예요.');
  data.subtext = str(data.subtext, '대화의 맥락과 말투를 바탕으로 해석한 결과예요.');

  const ti = data.timing_insight && typeof data.timing_insight === 'object' ? data.timing_insight : {};
  const speed = SPEEDS.includes(ti.reply_speed) ? ti.reply_speed : 'unknown';
  const analysis = str(ti.analysis);
  data.timing_insight = {
    available: ti.available === true && speed !== 'unknown' && !!analysis,
    reply_speed: speed,
    analysis,
  };

  const r = data.replies && typeof data.replies === 'object' ? data.replies : {};
  data.replies = {
    charm: str(r.charm, '오 그렇구나! 얘기해줘서 고마워 :)'),
    polite: str(r.polite, '알려줘서 고마워요. 편할 때 또 이야기해요.'),
    cut: str(r.cut, '확인했어요! 좋은 하루 보내세요.'),
  };
  data.warning = str(data.warning, '상대의 답장 속도에 맞춰 너무 몰아붙이지 않는 게 좋아요.');

  if (!data.is_valid_chat) {
    data.invalid_reason =
      data.invalid_reason || '대화 캡처가 아니거나 글자를 읽을 수 없어요. 말풍선이 잘 보이는 카톡·DM 캡처를 올려주세요.';
  }
  return data;
}

// ------------------------------------------------------------------
// Prompts
// ------------------------------------------------------------------
const COMMON_RULES = `
[공통 규칙]
- 보이는(주어진) 대화만 근거로 판단하고, 없는 사정이나 이전 대화를 지어내지 말 것.
- "~일 수 있어요", "~해 보여요"처럼 말하고 단정하지 말 것.
- 금지: 관계를 끝내라/고백해라/차단해라 같은 큰 결정 지시, 밀당·질투 유발·일부러 무시하기 같은 조종 전략, 상대 성격을 단정하거나 깎아내리는 표현.
- 친구 사이에서 웃음과 함께 오가는 거친 말은 장난일 가능성을 먼저 고려할 것.
- 모든 텍스트에 이모지 금지. 해시태그(status_tags)에는 # 기호 없이 단어만.
- MBTI는 답장 말투를 맞추는 데만 참고하고 속마음 판단 근거로 쓰지 말 것.

[점수 기준 temperature 0~100]
- 85~100: 먼저 질문·약속 제안, 즉각 답장, 긴 리액션, 적극적 관심
- 70~84: 부드러운 말투, 긍정적 맞장구, 비교적 빠른 답장
- 50~69: 예의 바른 단문, 질문은 적지만 친절함 유지
- 35~49: 단답 위주, 질문 없음, 늦은 답장, 대화를 마무리하려는 뉘앙스
- 0~34: 차가운 단답, 핑계나 거절, 형식적인 답장
- 같은 대화에는 항상 같은 기준으로 같은 점수를 줄 것.

[출력 항목]
- chat_type: 기본값은 'meaningful'.
  'everyday_routine'은 안부·밥·날씨·단순 일정 확인처럼 가볍고 감정 신호가 전혀 없는 대화에만 쓸 것.
  불만, 질책, 짜증, 서운함, 사과, 거절, 갈등, 부탁, 업무 지적, 늦은 시간 연락에 대한 불편함, 호감 표현이 하나라도 있으면 반드시 'meaningful'.
  직장 대화에서 상사·동료가 불편함이나 지적을 표현하면 'meaningful'이고 점수는 50 미만으로 줄 것.
- summary: 한 줄 결론, 35자 이내
- subtext: 상대 속마음 해석 2~3문장, 150자 이내
- status_tags: 상대 상태를 요약하는 단어 2~4개
- replies: 바로 보낼 수 있는 자연스러운 카톡 말투, 각 45자 이내
  * charm: 부담 없이 다정한 답장(과한 플러팅·고백 금지)
  * polite: 누구에게나 무난한 답장
  * cut: 정중하게 대화를 마무리하는 답장(차갑거나 공격적인 표현 금지)
- warning: 이 상황에서 조심할 점 한 줄, 60자 이내
- confidence: 상대 메시지가 3개 이하이거나 맥락이 애매하면 'low'
`;

function relationshipLine(relationship: string) {
  return relationship ? `- 사용자가 선택한 관계: ${relationship}` : '- 관계: 사용자가 선택하지 않음. 대화 내용으로 관계를 조심스럽게 짐작할 것.';
}
function mbtiLine(mbti: string) {
  return mbti ? `- 상대 MBTI(사용자 입력): ${mbti}` : '- 상대 MBTI: 모름';
}

function capturePrompt(relationship: string, mbti: string) {
  return `너는 한국어 메신저 대화를 읽고 상대의 속마음과 답장 타이밍을 해석하는 관계 코치야.
첨부 이미지를 보고 아래 순서대로 판단해서 지정된 JSON으로만 답해.

[1. 이미지 판별]
- 카카오톡, 인스타 DM, 문자 등 메신저 대화 캡처가 아니거나 글자를 읽을 수 없으면:
  is_valid_chat=false, invalid_reason에 이유 한 줄, 나머지는 빈 값이나 기본값.

[2. 화자 구분]
- 오른쪽 말풍선은 '나', 왼쪽 말풍선(프로필 사진이나 이름이 붙은 쪽)은 '상대'.
- 상대의 각 메시지는 바로 앞 내 메시지에 대한 반응으로 해석(답했는지, 화제를 이었는지, 바꿨는지, 질문했는지).
- 앞부분과 끝부분의 말투 온도 변화(존댓말↔반말, 웃음·이모티콘 감소, 마침표 단답)를 비교.

[3. 답장 텀 분석 timing_insight]
- 말풍선 옆에 찍힌 시각, 날짜 구분선, '읽음 1' 표시를 읽고 메시지 사이 시간 간격을 계산해.
- 시각이 보이면 available=true, reply_speed는 fast(대부분 몇 분 안) / moderate(수십 분~1시간) / slow(몇 시간 이상) / late_night(밤 11시~새벽 4시 대화 위주) / mixed(들쭉날쭉) 중 하나,
  analysis에는 상대의 답장 간격과 시간대가 무엇을 뜻하는지 1~2문장(90자 이내). 실제 시각을 근거로 쓸 것.
- 시각이 하나도 보이지 않으면 available=false, reply_speed='unknown', analysis=''. 절대 시간을 지어내지 말 것.

[4. 사용자 정보]
${relationshipLine(relationship)}
${mbtiLine(mbti)}
${COMMON_RULES}`;
}

function filePrompt(relationship: string, mbti: string, stats: any, transcript: string) {
  const s = stats || {};
  return `너는 한국어 카카오톡 대화를 읽고 상대의 속마음과 관계 흐름을 해석하는 관계 코치야.
브라우저에서 이름을 지우고 계산한 통계와 최근 대화를 보고 지정된 JSON으로만 답해. is_valid_chat은 true.

[대화 통계 - 코드로 정확히 계산된 값이니 다시 계산하지 말고 근거로 인용만 할 것]
- 메시지 수: 나 ${s.meCount ?? 0}개 / 상대 ${s.partnerCount ?? 0}개
- 6시간 이상 공백 뒤 먼저 말 건 비율: 나 ${s.starterRatio?.me ?? 50}% / 상대 ${s.starterRatio?.partner ?? 50}%
- 평균 답장 시간: 나 ${s.avgReplyMinutesMe ?? 0}분 / 상대 ${s.avgReplyMinutesPartner ?? 0}분
- 상대 메시지: 평균 ${s.partnerAvgLength ?? 0}자, 질문 비율 ${s.partnerQuestionRatio ?? 0}%, 웃음(ㅋ/ㅎ) 비율 ${s.partnerLaughRatio ?? 0}%
- 밤 11시~새벽 4시 대화 비율: ${s.lateNightRatio ?? 0}%
- 최근 30일 상대 메시지 추세: ${s.trend || 'flat'} (up=증가, down=감소, flat=비슷)

[답장 텀 분석 timing_insight]
- 위 평균 답장 시간과 밤 대화 비율을 근거로 available=true, reply_speed(fast/moderate/slow/late_night/mixed)를 정하고,
  analysis에 상대의 답장 간격이 무엇을 뜻하는지 1~2문장(90자 이내). 나와 상대의 답장 시간을 비교해 줄 것.

[사용자 정보]
${relationshipLine(relationship)}
${mbtiLine(mbti)}

[최근 대화 (이름은 '나', '상대'로 바뀜)]
${transcript}
${COMMON_RULES}`;
}

const responseSchema = {
  type: Type.OBJECT,
  properties: {
    is_valid_chat: { type: Type.BOOLEAN },
    invalid_reason: { type: Type.STRING },
    chat_type: { type: Type.STRING, enum: ['meaningful', 'everyday_routine'] },
    timing_insight: {
      type: Type.OBJECT,
      properties: {
        available: { type: Type.BOOLEAN },
        reply_speed: { type: Type.STRING, enum: SPEEDS },
        analysis: { type: Type.STRING },
      },
      required: ['available', 'reply_speed', 'analysis'],
    },
    temperature: { type: Type.INTEGER },
    confidence: { type: Type.STRING, enum: ['low', 'medium', 'high'] },
    summary: { type: Type.STRING },
    subtext: { type: Type.STRING },
    status_tags: { type: Type.ARRAY, items: { type: Type.STRING } },
    replies: {
      type: Type.OBJECT,
      properties: {
        charm: { type: Type.STRING },
        polite: { type: Type.STRING },
        cut: { type: Type.STRING },
      },
      required: ['charm', 'polite', 'cut'],
    },
    warning: { type: Type.STRING },
  },
  required: [
    'is_valid_chat', 'invalid_reason', 'chat_type', 'timing_insight', 'temperature',
    'confidence', 'summary', 'subtext', 'status_tags', 'replies', 'warning',
  ],
  propertyOrdering: [
    'is_valid_chat', 'invalid_reason', 'chat_type', 'timing_insight', 'temperature',
    'confidence', 'summary', 'subtext', 'status_tags', 'replies', 'warning',
  ],
};

// ------------------------------------------------------------------
// Analyze
// ------------------------------------------------------------------
// Tried in order. 'gemini-flash-latest' is Google's always-current alias, so at least one should exist.
const CANDIDATE_MODELS = ['gemini-flash-latest', 'gemini-3.8-flash', 'gemini-2.5-flash', 'gemini-flash-lite-latest'];
const ATTEMPT_TIMEOUT_MS = 25000;

app.post('/api/analyze', async (req, res) => {
  const startTime = Date.now();
  let executedModel = CANDIDATE_MODELS[0];
  try {
    const { mode = 'capture', imageBase64, mimeType, transcript, stats, customApiKey } = req.body || {};
    const relationship = typeof req.body?.relationship === 'string' ? req.body.relationship.slice(0, 20) : '';
    const mbti = typeof req.body?.mbti === 'string' && /^[EI][SN][TF][JP]$/.test(req.body.mbti) ? req.body.mbti : '';
    const apiKey = customApiKey || process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(400).json({ error: friendlyMessage('API_KEY_INVALID'), errorCode: 'API_KEY_MISSING' });
    }
    if (mode === 'capture' && !imageBase64) {
      return res.status(400).json({ error: '분석할 대화 캡처를 올려주세요.', errorCode: 'NO_IMAGE' });
    }
    if (mode === 'file' && (!transcript || !String(transcript).trim())) {
      return res.status(400).json({ error: '분석할 대화 내용이 비어 있어요.', errorCode: 'NO_TRANSCRIPT' });
    }

    const parts: any[] = [];
    if (mode === 'file') {
      parts.push({ text: filePrompt(relationship, mbti, stats, String(transcript).slice(0, 22000)) });
    } else {
      let data = String(imageBase64);
      let detectedMime = mimeType || 'image/jpeg';
      if (data.includes('base64,')) {
        const [head, body] = data.split('base64,');
        data = body;
        const m = head.match(/:(.*?);/);
        if (m) detectedMime = m[1];
      }
      parts.push({ inlineData: { mimeType: detectedMime, data } });
      parts.push({ text: capturePrompt(relationship, mbti) });
    }

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
    });

    let parsed: any = null;
    let lastErr: any = null;
    let lastCode = 'MODEL_ERROR';
    const MAX_ATTEMPTS = CANDIDATE_MODELS.length;

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      const model = CANDIDATE_MODELS[attempt];
      executedModel = model;
      const config: any = {
        temperature: 0.2,
        maxOutputTokens: 2048,
        responseMimeType: 'application/json',
        responseSchema,
      };
      // Turn thinking off on 2.5 models for speed (other models keep their default)
      if (model.startsWith('gemini-2.5')) config.thinkingConfig = { thinkingBudget: 0 };

      let timer: any;
      try {
        console.log(`[TALK-CATCH] attempt ${attempt + 1}/${MAX_ATTEMPTS} model=${model} mode=${mode}`);
        const timeout = new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error('TIMEOUT: attempt exceeded 25s')), ATTEMPT_TIMEOUT_MS);
        });
        const response: any = await Promise.race([
          ai.models.generateContent({ model, contents: [{ role: 'user', parts }], config }),
          timeout,
        ]);
        const text = response?.text || '';
        if (!text) throw new Error('MODEL_ERROR: empty response');
        parsed = cleanAndSanitizeJson(text);
        break;
      } catch (err: any) {
        lastErr = err;
        lastCode = getErrorCode(err);
        console.warn(`[TALK-CATCH] attempt ${attempt + 1} failed [${lastCode}] model=${model}: ${err?.message || err}`);
        if (lastCode === 'API_KEY_INVALID') break;
        if (attempt < MAX_ATTEMPTS - 1) await new Promise((r) => setTimeout(r, lastCode === 'RATE_LIMIT' ? 2000 : 500));
      } finally {
        clearTimeout(timer);
      }
    }

    if (!parsed) {
      const code = lastCode;
      console.error(`[TALK-CATCH] all attempts failed [${code}] last model=${executedModel}`);
      const detail = `${executedModel}: ${String(lastErr?.message || lastErr || '').slice(0, 200)}`;
      return res.status(code === 'RATE_LIMIT' ? 429 : 502).json({ error: friendlyMessage(code), errorCode: code, detail });
    }

    parsed.elapsed_ms = Date.now() - startTime;
    parsed.model = executedModel;
    console.log(`[TALK-CATCH] success mode=${mode} model=${executedModel} ${parsed.elapsed_ms}ms`);
    return res.json(parsed);
  } catch (err: any) {
    const code = getErrorCode(err);
    console.error(`[TALK-CATCH] unexpected failure [${code}] model=${executedModel}:`, err?.message || err);
    return res.status(500).json({ error: friendlyMessage(code), errorCode: code, detail: String(err?.message || err || '').slice(0, 200) });
  }
});

async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';
  const port = Number(process.env.PORT) || 3000;

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`[TALK-CATCH] Server started on http://0.0.0.0:${port}`);
  });
}

startServer();
