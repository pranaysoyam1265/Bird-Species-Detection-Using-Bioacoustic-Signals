/**
 * Avatar store — persists a user-uploaded avatar as a data-URI in localStorage.
 * Max size is ~400 KB to stay within typical localStorage budgets.
 */

const AVATAR_KEY = "birdsense-avatar"
const MAX_SIZE_PX = 256 // resize uploaded images to this square

/**
 * Read the saved avatar data-URI (or null if none).
 */
export function getAvatar(): string | null {
  try {
    return localStorage.getItem(AVATAR_KEY)
  } catch {
    return null
  }
}

/**
 * Delete the saved avatar.
 */
export function clearAvatar(): void {
  try {
    localStorage.removeItem(AVATAR_KEY)
  } catch { /* */ }
}

/**
 * Process an image File, resize it to a square, and save as a JPEG data-URI.
 * Returns the data-URI on success.
 */
export function saveAvatarFromFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error("Failed to read file"))
    reader.onload = () => {
      const img = new Image()
      img.onerror = () => reject(new Error("Invalid image"))
      img.onload = () => {
        // Draw to canvas, center-crop to square
        const canvas = document.createElement("canvas")
        canvas.width = MAX_SIZE_PX
        canvas.height = MAX_SIZE_PX
        const ctx = canvas.getContext("2d")!
        const min = Math.min(img.width, img.height)
        const sx = (img.width - min) / 2
        const sy = (img.height - min) / 2
        ctx.drawImage(img, sx, sy, min, min, 0, 0, MAX_SIZE_PX, MAX_SIZE_PX)
        const dataUri = canvas.toDataURL("image/jpeg", 0.8)
        try {
          localStorage.setItem(AVATAR_KEY, dataUri)
        } catch {
          reject(new Error("Storage full"))
          return
        }
        resolve(dataUri)
      }
      img.src = reader.result as string
    }
    reader.readAsDataURL(file)
  })
}
