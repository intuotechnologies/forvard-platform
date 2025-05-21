# 1. Base image for building the application
FROM node:20-alpine AS builder

# 2. Set working directory
WORKDIR /app

# 3. Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# 4. Install dependencies
# Use --ci for reproducible builds, ensuring it uses package-lock.json
RUN npm ci

# 5. Copy the rest of the application source code
COPY . .

# 6. Build the Next.js application
# This command typically comes from your package.json scripts
RUN npm run build

# 7. Production image: Use a lean base image
FROM node:20-alpine AS runner

WORKDIR /app

# Copy environment configuration - Next.js doesn't bundle .env files by default
# You might need to handle .env files or runtime environment variables differently in your deployment
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./
# If your Next.js app needs node_modules in production for some reason (usually not for standalone output)
# COPY --from=builder /app/node_modules ./node_modules

# 8. Expose the port the app runs on
EXPOSE 3000

# 9. Command to run the application
# This command typically comes from your package.json scripts for starting in production mode
CMD ["npm", "start"] 