import { z } from "zod"

export const loginSchema = z.object({
  email: z
    .string()
    .min(1, "EMAIL_REQUIRED")
    .email("INVALID_EMAIL_FORMAT"),
  password: z
    .string()
    .min(6, "PASSWORD_MIN_6_CHARS"),
})

export const signupSchema = z
  .object({
    email: z
      .string()
      .min(1, "EMAIL_REQUIRED")
      .email("INVALID_EMAIL_FORMAT"),
    password: z
      .string()
      .min(6, "PASSWORD_MIN_6_CHARS")
      .max(100, "PASSWORD_TOO_LONG"),
    confirmPassword: z.string(),
    name: z.string().optional(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "PASSWORDS_DO_NOT_MATCH",
    path: ["confirmPassword"],
  })

export type LoginInput = z.infer<typeof loginSchema>
export type SignupInput = z.infer<typeof signupSchema>
