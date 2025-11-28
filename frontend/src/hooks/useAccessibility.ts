import React, { useEffect, useRef } from 'react';

interface AccessibilityAnnouncer {
  announce: (message: string) => void;
}

const useAccessibility = (): AccessibilityAnnouncer => {
  const [announcement, setAnnouncement] = useState<string | null>(null);
  const [isScreenReaderEnabled, setIsScreenReaderEnabled] = useState(false);

  const announce = (message: string) => {
    setAnnouncement(message);

    // Clear announcement after 5 seconds
    setTimeout(() => {
      setAnnouncement(null);
    }, 5000);
  };

  // Check for screen reader
  useEffect(() => {
    const checkScreenReader = () => {
      try {
        const hasScreenReader = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        setIsScreenReaderEnabled(hasScreenReader);
      } catch (error) {
        console.error('Error checking screen reader:', error);
      }
    };

    checkScreenReader();
  }, []);

  return {
    announce,
    announcement,
    isScreenReaderEnabled,
  };
};

export default useAccessibility;
