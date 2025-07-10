# Stage 1: Build the application
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json from the frontend directory
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application source code from the frontend directory
COPY frontend/ .

# Build the Next.js application.
# No build-time env vars needed for the proxy anymore.
RUN npm run build

# Stage 2: Production image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Create a non-root user for security
RUN addgroup -S nextjs && adduser -S nextjs -G nextjs

# Copy dependency files from the builder stage
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json

# Install only production dependencies
RUN npm ci --omit=dev

# Copy built assets from the builder stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts

# Change ownership of all files to the non-root user
RUN chown -R nextjs:nextjs /app

# Switch to the non-root user
USER nextjs

# Set the runtime environment variable for the API URL
# This is now a server-side only variable, read at runtime by the API proxy.
ENV API_URL="http://forvard_api:8443"

# Expose the port the app runs on
EXPOSE 3000

# The command to start the application
CMD ["npm", "start"] 