/**
 * VirtualList 包装器
 * 为大列表提供虚拟滚动能力，减少渲染项目数
 */

import React, { useMemo } from 'react';
import styled from 'styled-components';
import { spacing } from '../../styles/tokens';

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  containerHeight?: number;
  gap?: number;
}

const Container = styled.div`
  position: relative;
  overflow-y: auto;
  overflow-x: hidden;
`;

const Viewport = styled.div`
  position: relative;
`;

const ItemWrapper = styled.div`
  position: absolute;
  left: 0;
  right: 0;
`;

/**
 * 简单虚拟列表实现
 * 只渲染可见区域内的项目 + 缓冲区
 */
export function VirtualList<T>({
  items,
  itemHeight,
  renderItem,
  containerHeight = 400,
  gap = spacing.md,
}: VirtualListProps<T>) {
  const totalHeight = items.length * (itemHeight + gap);
  const [scrollTop, setScrollTop] = React.useState(0);

  // 计算可见范围
  const startIndex = Math.max(0, Math.floor(scrollTop / (itemHeight + gap)) - 2); // 缓冲 2 个
  const endIndex = Math.min(items.length, Math.ceil((scrollTop + containerHeight) / (itemHeight + gap)) + 2);

  const visibleItems = useMemo(
    () => items.slice(startIndex, endIndex),
    [items, startIndex, endIndex]
  );

  const offsetY = startIndex * (itemHeight + gap);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop((e.target as HTMLDivElement).scrollTop);
  };

  return (
    <Container
      style={{ height: containerHeight }}
      onScroll={handleScroll}
    >
      <Viewport style={{ height: totalHeight }}>
        {visibleItems.map((item, i) => (
          <ItemWrapper
            key={startIndex + i}
            style={{
              top: offsetY + i * (itemHeight + gap),
              height: itemHeight,
            }}
          >
            {renderItem(item, startIndex + i)}
          </ItemWrapper>
        ))}
      </Viewport>
    </Container>
  );
}

/**
 * 轻量虚拟表格（基于 Ant Table 但添加虚拟滚动）
 */
export function VirtualTable<T extends Record<string, any>>({
  columns,
  dataSource,
  itemHeight = 48,
  containerHeight = 600,
}: {
  columns: Array<{ title: string; dataIndex: string; key: string; width?: number }>;
  dataSource: T[];
  itemHeight?: number;
  containerHeight?: number;
}) {
  const [scrollTop, setScrollTop] = React.useState(0);

  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - 2);
  const endIndex = Math.min(dataSource.length, Math.ceil((scrollTop + containerHeight) / itemHeight) + 2);
  const visibleRows = dataSource.slice(startIndex, endIndex);

  const offsetY = startIndex * itemHeight;
  const totalHeight = dataSource.length * itemHeight;

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop((e.target as HTMLDivElement).scrollTop);
  };

  return (
    <Container style={{ height: containerHeight }} onScroll={handleScroll}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key} style={{ width: col.width, textAlign: 'left', padding: 8 }}>
                {col.title}
              </th>
            ))}
          </tr>
        </thead>
      </table>
      <Viewport style={{ height: totalHeight }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <tbody>
            {visibleRows.map((row, i) => (
              <tr
                key={startIndex + i}
                style={{
                  position: 'absolute',
                  top: offsetY + i * itemHeight,
                  left: 0,
                  right: 0,
                  height: itemHeight,
                  borderBottom: '1px solid var(--border-default)',
                }}
              >
                {columns.map((col) => (
                  <td key={col.key} style={{ width: col.width, padding: 8 }}>
                    {row[col.dataIndex]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Viewport>
    </Container>
  );
}

export default VirtualList;
