export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // ── Proxy route ────────────────────────────────────────────────────────
    if (url.pathname === '/api/chat') {

      // Only POST allowed
      if (request.method === 'OPTIONS') {
        return cors(null, 204, request);
      }
      if (request.method !== 'POST') {
        return cors(new Response('Method Not Allowed', { status: 405 }), 405, request);
      }

      // Only allow requests from pesnik.dev itself
      const origin = request.headers.get('Origin') || '';
      const allowed = ['https://pesnik.dev', 'https://www.pesnik.dev'];
      if (!allowed.includes(origin)) {
        return new Response('Forbidden', { status: 403 });
      }

      let body;
      try {
        body = await request.json();
      } catch {
        return cors(new Response('Bad Request', { status: 400 }), 400, request);
      }

      // Forward to OpenWebUI with secret key from env
      let upstream;
      try {
        upstream = await fetch('https://ai.pesnik.dev/api/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${env.OPENWEBUI_API_KEY}`,
          },
          body: JSON.stringify(body),
        });
      } catch (err) {
        return cors(
          new Response(JSON.stringify({ error: err.message }), { status: 502 }),
          502, request
        );
      }

      const data = await upstream.text();
      return cors(
        new Response(data, {
          status: upstream.status,
          headers: { 'Content-Type': 'application/json' },
        }),
        upstream.status, request
      );
    }

    // ── Everything else → static assets ───────────────────────────────────
    return env.ASSETS.fetch(request);
  },
};

function cors(response, status, request) {
  const origin = request.headers.get('Origin') || '';
  const headers = {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
  };

  if (response === null) {
    return new Response(null, { status, headers });
  }

  const res = new Response(response.body, response);
  Object.entries(headers).forEach(([k, v]) => res.headers.set(k, v));
  return res;
}
