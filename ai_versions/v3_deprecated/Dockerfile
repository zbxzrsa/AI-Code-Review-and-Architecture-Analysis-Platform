FROM alpine:3.18

WORKDIR /archive

# Just store the deprecated version - no runtime
COPY metrics.json .
COPY config.yaml .

# Add metadata about why it failed
RUN echo "Deprecated: High latency (12.4s) and cost ($0.05 per review)" > failure_reason.txt

# This version is archived for reference only
CMD ["tail", "-f", "/dev/null"]