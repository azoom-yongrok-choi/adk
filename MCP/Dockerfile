# MCP/Dockerfile
FROM alpine:3.19

WORKDIR /app

# Toolbox 바이너리 다운로드
RUN apk add --no-cache curl && \
    curl -O https://storage.googleapis.com/genai-toolbox/v0.6.0/linux/amd64/toolbox && \
    chmod +x toolbox

COPY tools.yaml .

EXPOSE 5001
CMD ["./toolbox", "--tools-file", "tools.yaml", "-p", "5001", "-a", "0.0.0.0"]