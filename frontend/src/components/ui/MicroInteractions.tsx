import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button, Tooltip, Badge, notification } from 'antd';
import { CheckOutlined, LoadingOutlined } from '@ant-design/icons';

// Loading button with animation and success feedback
export const LoadingButton: React.FC<{
  onClick: () => Promise<any>;
  children: React.ReactNode;
  className?: string;
  type?: 'primary' | 'default' | 'dashed' | 'link' | 'text';
}> = ({ onClick, children, className = '', type = 'primary' }) => {
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleClick = async () => {
    setLoading(true);
    try {
      await onClick();
      setSuccess(true);
      setTimeout(() => setSuccess(false), 2000);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button
      type={type}
      className={`micro-interaction-button ${className}`}
      onClick={handleClick}
      disabled={loading}
      icon={
        loading ? (
          <LoadingOutlined />
        ) : success ? (
          <CheckOutlined style={{ color: '#52c41a' }} />
        ) : null
      }
    >
      {children}
    </Button>
  );
};

// Pulse effect to draw attention
export const PulseEffect: React.FC<{
  children: React.ReactNode;
  active?: boolean;
  color?: string;
}> = ({ children, active = true, color = '#1677ff' }) => {
  return (
    <div className="pulse-container" style={{ position: 'relative' }}>
      {active && (
        <div
          className="pulse-ring"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            borderRadius: 'inherit',
            animation: 'pulse 1.5s infinite',
          }}
        />
      )}
      <style>{`
        @keyframes pulse {
          0% {
            box-shadow: 0 0 0 0 ${color}40;
          }
          70% {
            box-shadow: 0 0 0 10px ${color}00;
          }
          100% {
            box-shadow: 0 0 0 0 ${color}00;
          }
        }
      `}</style>
      {children}
    </div>
  );
};

// Fade-in component for smooth appearance
export const FadeIn: React.FC<{
  children: React.ReactNode;
  delay?: number;
  duration?: number;
}> = ({ children, delay = 0, duration = 0.5 }) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay, duration }}
    >
      {children}
    </motion.div>
  );
};

// Slide-in component for smooth entrance
export const SlideIn: React.FC<{
  children: React.ReactNode;
  direction?: 'left' | 'right' | 'up' | 'down';
  delay?: number;
  duration?: number;
}> = ({ children, direction = 'up', delay = 0, duration = 0.5 }) => {
  const directionMap = {
    left: { x: -20, y: 0 },
    right: { x: 20, y: 0 },
    up: { x: 0, y: 20 },
    down: { x: 0, y: -20 },
  };

  return (
    <motion.div
      initial={{ opacity: 0, ...directionMap[direction] }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ delay, duration }}
    >
      {children}
    </motion.div>
  );
};

// Hover effect for cards and elements
export const HoverEffect: React.FC<{
  children: React.ReactNode;
  scale?: number;
  elevation?: boolean;
}> = ({ children, scale = 1.02, elevation = true }) => {
  return (
    <motion.div
      whileHover={{
        scale,
        boxShadow: elevation
          ? '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
          : undefined,
      }}
      transition={{ duration: 0.2 }}
    >
      {children}
    </motion.div>
  );
};

// Success animation for positive feedback
export const SuccessAnimation: React.FC<{
  show: boolean;
  onComplete?: () => void;
}> = ({ show, onComplete }) => {
  useEffect(() => {
    if (show) {
      const timer = setTimeout(() => {
        onComplete && onComplete();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [show, onComplete]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 1.5, opacity: 0 }}
          transition={{ duration: 0.5 }}
          style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1000,
            background: 'rgba(82, 196, 26, 0.9)',
            borderRadius: '50%',
            width: 80,
            height: 80,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <CheckOutlined style={{ fontSize: 40, color: 'white' }} />
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Animated badge to show new content or notifications
export const AnimatedBadge: React.FC<{
  count: number;
  children: React.ReactNode;
  dot?: boolean;
}> = ({ count, children, dot = false }) => {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    if (count > 0) {
      setAnimated(true);
      const timer = setTimeout(() => setAnimated(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [count]);

  return (
    <Badge
      count={count}
      dot={dot}
      className={animated ? 'badge-animated' : ''}
      style={{ position: 'relative' }}
    >
      <style>{`
        .badge-animated::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          background-color: #ff4d4f;
          animation: badgePulse 1.5s infinite;
          z-index: -1;
        }
        
        @keyframes badgePulse {
          0% {
            transform: scale(1);
            opacity: 0.7;
          }
          100% {
            transform: scale(2);
            opacity: 0;
          }
        }
      `}</style>
      {children}
    </Badge>
  );
};

// Enhanced tooltip with animations
export const EnhancedTooltip: React.FC<{
  title: React.ReactNode;
  children: React.ReactNode;
  placement?: 'top' | 'left' | 'right' | 'bottom';
}> = ({ title, children, placement = 'top' }) => {
  return (
    <Tooltip
      title={
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {title}
        </motion.div>
      }
      placement={placement}
    >
      {children}
    </Tooltip>
  );
};

// Success notification helper
export const showSuccessNotification = (
  message: string,
  description?: string
) => {
  notification.success({
    message,
    description,
    icon: (
      <motion.div
        initial={{ rotate: -90, scale: 0.5 }}
        animate={{ rotate: 0, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <CheckOutlined style={{ color: '#52c41a' }} />
      </motion.div>
    ),
  });
};

// Export all micro interactions
export default {
  LoadingButton,
  PulseEffect,
  FadeIn,
  SlideIn,
  HoverEffect,
  SuccessAnimation,
  AnimatedBadge,
  EnhancedTooltip,
  showSuccessNotification,
};
