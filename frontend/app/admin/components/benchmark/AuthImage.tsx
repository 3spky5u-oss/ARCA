'use client';

import React, { useEffect, useState } from 'react';

/**
 * Authenticated image component - fetches images via auth headers
 * so chart endpoints behind verify_admin work correctly.
 * Browser <img src> tags do not send Authorization headers,
 * so we fetch as blob and use an object URL instead.
 */
export function AuthImage({
  src,
  alt,
  className,
  loading,
  onClick,
}: {
  src: string;
  alt: string;
  className?: string;
  loading?: 'lazy' | 'eager';
  onClick?: () => void;
}) {
  const [loadedImage, setLoadedImage] = useState<{
    src: string;
    objectUrl: string | null;
    error: boolean;
  }>({
    src: '',
    objectUrl: null,
    error: false,
  });

  const objectUrl = loadedImage.src === src ? loadedImage.objectUrl : null;
  const error = loadedImage.src === src ? loadedImage.error : false;

  useEffect(() => {
    let cancelled = false;

    const token =
      (typeof window !== 'undefined' && localStorage.getItem('arca_auth_token')) ||
      (typeof window !== 'undefined' && sessionStorage.getItem('arca_admin_token')) ||
      '';

    fetch(src, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;

        const nextObjectUrl = URL.createObjectURL(blob);
        setLoadedImage((prev) => {
          if (prev.objectUrl && prev.objectUrl !== nextObjectUrl) {
            URL.revokeObjectURL(prev.objectUrl);
          }
          return { src, objectUrl: nextObjectUrl, error: false };
        });
      })
      .catch(() => {
        if (cancelled) return;

        setLoadedImage((prev) => {
          if (prev.objectUrl) {
            URL.revokeObjectURL(prev.objectUrl);
          }
          return { src, objectUrl: null, error: true };
        });
      });

    return () => {
      cancelled = true;
    };
  }, [src]);

  useEffect(() => {
    return () => {
      if (loadedImage.objectUrl) {
        URL.revokeObjectURL(loadedImage.objectUrl);
      }
    };
  }, [loadedImage.objectUrl]);

  if (error) {
    return (
      <div className="text-gray-500 text-sm py-8 text-center">
        Chart unavailable
      </div>
    );
  }

  if (!objectUrl) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="w-5 h-5 border-2 border-gray-600 border-t-blue-400 rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <img
      src={objectUrl}
      alt={alt}
      className={className}
      loading={loading}
      onClick={onClick}
    />
  );
}
