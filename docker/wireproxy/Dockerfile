FROM alpine:3.21 AS fetch
ARG VERSION=v0.4.2
ARG TARGETARCH=amd64
RUN apk add --no-cache wget ca-certificates \
    && wget -qO /tmp/wireproxy.tar.gz \
       "https://github.com/pufferffish/wireproxy/releases/download/${VERSION}/wireproxy_linux_${TARGETARCH}.tar.gz" \
    && tar -xzf /tmp/wireproxy.tar.gz -C /usr/local/bin wireproxy \
    && chmod +x /usr/local/bin/wireproxy \
    && rm /tmp/wireproxy.tar.gz

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=fetch /usr/local/bin/wireproxy /usr/local/bin/wireproxy
ENTRYPOINT ["/usr/local/bin/wireproxy"]
