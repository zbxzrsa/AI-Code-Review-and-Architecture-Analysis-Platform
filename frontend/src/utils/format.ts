/**
 * 格式化工具函数
 * 提供日期、数字、文本等格式化功能
 */

// 日期格式化
export const formatDate = (date: string | Date, format: string = 'YYYY-MM-DD'): string => {
  const d = new Date(date);
  
  if (isNaN(d.getTime())) {
    return '无效日期';
  }

  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const hours = String(d.getHours()).padStart(2, '0');
  const minutes = String(d.getMinutes()).padStart(2, '0');
  const seconds = String(d.getSeconds()).padStart(2, '0');

  return format
    .replace('YYYY', String(year))
    .replace('MM', month)
    .replace('DD', day)
    .replace('HH', hours)
    .replace('mm', minutes)
    .replace('ss', seconds);
};

// 相对时间格式化
export const formatRelativeTime = (date: string | Date): string => {
  const now = new Date();
  const target = new Date(date);
  const diffMs = now.getTime() - target.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSeconds < 60) {
    return '刚刚';
  } else if (diffMinutes < 60) {
    return `${diffMinutes}分钟前`;
  } else if (diffHours < 24) {
    return `${diffHours}小时前`;
  } else if (diffDays < 7) {
    return `${diffDays}天前`;
  } else {
    return formatDate(date);
  }
};

// 文件大小格式化
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

// 数字格式化（千分位）
export const formatNumber = (num: number): string => {
  return num.toLocaleString('zh-CN');
};

// 百分比格式化
export const formatPercentage = (value: number, decimals: number = 1): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

// 代码行数格式化
export const formatLineCount = (lines: number): string => {
  if (lines < 1000) {
    return `${lines} 行`;
  } else if (lines < 1000000) {
    return `${(lines / 1000).toFixed(1)}K 行`;
  } else {
    return `${(lines / 1000000).toFixed(1)}M 行`;
  }
};

// 复杂度等级格式化
export const formatComplexityLevel = (complexity: number): { level: string; color: string } => {
  if (complexity <= 5) {
    return { level: '简单', color: 'green' };
  } else if (complexity <= 10) {
    return { level: '中等', color: 'orange' };
  } else if (complexity <= 20) {
    return { level: '复杂', color: 'red' };
  } else {
    return { level: '极复杂', color: 'darkred' };
  }
};

// 安全等级格式化
export const formatSecurityLevel = (level: 'low' | 'medium' | 'high' | 'critical'): { text: string; color: string } => {
  const levels = {
    low: { text: '低风险', color: 'green' },
    medium: { text: '中风险', color: 'orange' },
    high: { text: '高风险', color: 'red' },
    critical: { text: '严重', color: 'darkred' }
  };
  
  return levels[level] || { text: '未知', color: 'gray' };
};

// 截断文本
export const truncateText = (text: string, maxLength: number = 100): string => {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.substring(0, maxLength)}...`;
};

// 高亮搜索关键词
export const highlightSearchTerm = (text: string, searchTerm: string): string => {
  if (!searchTerm) return text;
  
  const regex = new RegExp(`(${searchTerm})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
};