export function Logo({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 128 128" aria-hidden="true">
      <rect width="128" height="128" rx="28" fill="#14213D" />
      <path d="M30 30h-8v68h8" fill="none" stroke="#F2A541" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M98 30h8v68h-8" fill="none" stroke="#F2A541" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M44 92V36l40 56V36" fill="none" stroke="#F8F5EE" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
