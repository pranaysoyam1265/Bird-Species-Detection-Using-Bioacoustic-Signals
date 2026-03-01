/** @type {import('next').NextConfig} */
const nextConfig = {
  // Removed output: export for API route support on Vercel
  // output: 'export', // Commented out for API route compatibility with Vercel
  serverExternalPackages: ['better-sqlite3'],
  typescript: {
    ignoreBuildErrors: true,
  }
}

export default nextConfig
