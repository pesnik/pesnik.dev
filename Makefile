.PHONY: deploy secret dev

deploy:
	npx wrangler deploy worker.js

secret:
	npx wrangler secret put OPENWEBUI_API_KEY

dev:
	npx wrangler dev worker.js

tail:
	npx wrangler tail
