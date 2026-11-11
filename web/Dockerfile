# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-alpine AS build
WORKDIR /app
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
RUN addgroup -S niyam && adduser -S niyam -G niyam
COPY --from=build --chown=niyam:niyam /app/.next/standalone ./
COPY --from=build --chown=niyam:niyam /app/.next/static ./.next/static
COPY --from=build --chown=niyam:niyam /app/public ./public
USER niyam
EXPOSE 3000
CMD ["node", "server.js"]
