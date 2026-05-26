FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

ARG VITE_ENDPOINT_PIPELINE=/api/pipeline/frame
ARG VITE_ENDPOINT_DETECT=/api/detection/detect
ARG VITE_ENDPOINT_EMOTION=/api/emotion/
ARG VITE_ENDPOINT_SPOOF=/api/anti-spoofing/anti-spoof
ARG VITE_ENDPOINT_VERIFY=/api/verification/verify
ARG VITE_ENDPOINT_REGISTER=/api/verification/register
ARG VITE_FRAME_INTERVAL_MS=2000

ENV VITE_ENDPOINT_PIPELINE=$VITE_ENDPOINT_PIPELINE \
    VITE_ENDPOINT_DETECT=$VITE_ENDPOINT_DETECT \
    VITE_ENDPOINT_EMOTION=$VITE_ENDPOINT_EMOTION \
    VITE_ENDPOINT_SPOOF=$VITE_ENDPOINT_SPOOF \
    VITE_ENDPOINT_VERIFY=$VITE_ENDPOINT_VERIFY \
    VITE_ENDPOINT_REGISTER=$VITE_ENDPOINT_REGISTER \
    VITE_FRAME_INTERVAL_MS=$VITE_FRAME_INTERVAL_MS

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
