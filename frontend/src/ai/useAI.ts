import { reviewInBrowser } from './webllmClient';

export async function review(text: string, channel: 'stable' | 'next' | 'legacy' = 'stable') {
  if (channel === 'next') return reviewInBrowser(text);
  const res = await fetch('/api/ai/review', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-AI-Channel': channel,
    },
    body: JSON.stringify({ text }),
  });
  return (await res.json()).result;
}
