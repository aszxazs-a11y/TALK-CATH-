// Vercel serverless function: GET /api/health
export default function handler(_req: any, res: any) {
  res.status(200).json({ status: 'ok', hasServerKey: !!process.env.GEMINI_API_KEY, time: new Date().toISOString() });
}
