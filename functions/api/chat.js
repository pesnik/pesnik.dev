export async function onRequestPost(context) {
  const { request, env } = context;

  // CORS — only allow requests from your own domain
  const origin = request.headers.get('Origin') || '';
  const allowed = ['https://pesnik.dev', 'https://www.pesnik.dev'];
  if (!allowed.includes(origin)) {
    return new Response('Forbidden', { status: 403 });
  }

  // Parse the incoming body from the frontend
  let body;
  try {
    body = await request.json();
  } catch {
    return new Response('Bad Request', { status: 400 });
  }

  // Basic rate-limit via Cloudflare's CF-Connecting-IP header
  // (Cloudflare KV-based rate limiting is optional — see below)
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';

  // Forward to OpenWebUI with the secret key from env
  const upstream = await fetch('https://ai.pesnik.dev/api/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${env.OPENWEBUI_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  const data = await upstream.json();

  return new Response(JSON.stringify(data), {
    status: upstream.status,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'POST',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

// Handle preflight OPTIONS
export async function onRequestOptions(context) {
  const origin = context.request.headers.get('Origin') || '';
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400',
    },
  });
}
