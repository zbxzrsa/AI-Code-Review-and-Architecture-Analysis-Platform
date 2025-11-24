/**
 * 前端代码分割和虚拟滚动优化
 * 实现懒加载、代码分割和虚拟化列表以提升性能
 */

import React, { Suspense, lazy, ComponentType } from 'react';
import { Spin } from 'antd';

// ============ 代码分割和懒加载 ============

/**
 * 懒加载组件包装器
 */
export function lazyLoad<T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>,
  fallback?: ComponentType<any>
) {
  const LazyComponent = lazy(importFunc);

  return (props: React.ComponentProps<T>) => (
    <Suspense fallback={fallback ? <fallback /> : <LoadingSpinner />}>
      <LazyComponent {...props} />
    </Suspense>
  );
}

/**
 * 页面级懒加载
 */
export const lazyLoadPage = <T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>
) => {
  return lazyLoad(importFunc, PageLoadingFallback);
};

/**
 * 组件级懒加载
 */
export const lazyLoadComponent = <T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>
) => {
  return lazyLoad(importFunc, ComponentLoadingFallback);
};

// ============ 加载状态组件 ============

/**
 * 页面加载占位符
 */
export const PageLoadingFallback: React.FC = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '400px',
    flexDirection: 'column'
  }}>
    <Spin size="large" />
    <div style={{ marginTop: 16, color: '#666' }}>
      Loading page...
    </div>
  </div>
);

/**
 * 组件加载占位符
 */
export const ComponentLoadingFallback: React.FC = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '20px'
  }}>
    <Spin size="small" />
  </div>
);

/**
 * 简单加载指示器
 */
export const LoadingSpinner: React.FC = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '20px'
  }}>
    <Spin />
  </div>
);

// ============ 虚拟化列表 ============

interface VirtualListItem {
  id: string | number;
  [key: string]: any;
}

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number | ((item: T, index: number) => number);
  renderItem: (item: T, index: number) => React.ReactNode;
  containerHeight?: number;
  overscanCount?: number;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * 简单的虚拟化列表实现
 */
export const VirtualList = <T extends VirtualListItem>({
  items,
  itemHeight,
  renderItem,
  containerHeight = 400,
  overscanCount = 5,
  className = '',
  style = {}
}: VirtualListProps<T>) => {
  const [scrollTop, setScrollTop] = React.useState(0);

  const getItemHeight = React.useCallback((item: T, index: number) => {
    return typeof itemHeight === 'function' ? itemHeight(item, index) : itemHeight;
  }, [itemHeight]);

  const visibleItems = React.useMemo(() => {
    let startIndex = 0;
    let endIndex = items.length;
    let accumulatedHeight = 0;

    // 计算可见范围
    for (let i = 0; i < items.length; i++) {
      const height = getItemHeight(items[i], i);
      if (accumulatedHeight + height < scrollTop) {
        startIndex = i + 1;
      } else if (accumulatedHeight > scrollTop + containerHeight) {
        endIndex = i;
        break;
      }
      accumulatedHeight += height;
    }

    return items.slice(startIndex, endIndex);
  }, [items, scrollTop, getItemHeight, containerHeight]);

  const totalHeight = React.useMemo(() => {
    return items.reduce((total, item, index) => {
      return total + getItemHeight(item, index);
    }, 0);
  }, [items, getItemHeight]);

  const handleScroll = React.useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  return (
    <div
      className={className}
      style={{
        height: containerHeight,
        overflow: 'auto',
        ...style
      }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${scrollTop}px)` }}>
          {visibleItems.map((item, index) => (
            <div
              key={item.id}
              style={{
                height: getItemHeight(item, index),
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0
              }}
            >
              {renderItem(item, index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default {
  lazyLoad,
  lazyLoadPage,
  lazyLoadComponent,
  PageLoadingFallback,
  ComponentLoadingFallback,
  LoadingSpinner,
  VirtualList
};
