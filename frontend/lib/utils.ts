import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a number with commas
 */
export function formatNumber(num: number): string {
  return new Intl.NumberFormat().format(num);
}

/**
 * Format a percentage
 */
export function formatPercent(num: number, decimals = 1): string {
  return `${num.toFixed(decimals)}%`;
}

/**
 * Format an exceedance factor (e.g., 1.44x)
 */
export function formatFactor(num: number): string {
  return `${num.toFixed(2)}x`;
}
