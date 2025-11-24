/**
 * 格式化工具函数单元测试
 */
import {
  formatDate,
  formatRelativeTime,
  formatFileSize,
  formatNumber,
  formatPercentage,
  formatLineCount,
  formatComplexityLevel,
  formatSecurityLevel,
  truncateText,
  highlightSearchTerm
} from '../format';

describe('格式化工具函数测试', () => {
  describe('formatDate', () => {
    test('应该正确格式化日期字符串', () => {
      const date = '2023-05-15T10:30:45Z';
      expect(formatDate(date)).toBe('2023-05-15');
    });

    test('应该正确格式化Date对象', () => {
      const date = new Date('2023-05-15T10:30:45Z');
      expect(formatDate(date)).toBe('2023-05-15');
    });

    test('应该支持自定义格式', () => {
      const date = '2023-05-15T10:30:45Z';
      expect(formatDate(date, 'YYYY-MM-DD HH:mm:ss')).toBe('2023-05-15 10:30:45');
    });

    test('应该处理无效日期', () => {
      expect(formatDate('invalid-date')).toBe('无效日期');
    });
  });

  describe('formatRelativeTime', () => {
    beforeEach(() => {
      // Mock Date.now() 为固定时间
      jest.spyOn(Date, 'now').mockImplementation(() => new Date('2023-05-15T12:00:00Z').getTime());
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    test('应该显示"刚刚"对于30秒内的时间', () => {
      const date = new Date('2023-05-15T11:59:30Z');
      expect(formatRelativeTime(date)).toBe('刚刚');
    });

    test('应该显示分钟数对于1小时内的时间', () => {
      const date = new Date('2023-05-15T11:30:00Z');
      expect(formatRelativeTime(date)).toBe('30分钟前');
    });

    test('应该显示小时数对于24小时内的时间', () => {
      const date = new Date('2023-05-15T08:00:00Z');
      expect(formatRelativeTime(date)).toBe('4小时前');
    });

    test('应该显示天数对于7天内的时间', () => {
      const date = new Date('2023-05-13T12:00:00Z');
      expect(formatRelativeTime(date)).toBe('2天前');
    });

    test('应该显示完整日期对于7天以上的时间', () => {
      const date = new Date('2023-05-01T12:00:00Z');
      expect(formatRelativeTime(date)).toBe('2023-05-01');
    });
  });

  describe('formatFileSize', () => {
    test('应该正确格式化字节数', () => {
      expect(formatFileSize(0)).toBe('0 B');
      expect(formatFileSize(512)).toBe('512 B');
      expect(formatFileSize(1024)).toBe('1 KB');
      expect(formatFileSize(1536)).toBe('1.5 KB');
      expect(formatFileSize(1048576)).toBe('1 MB');
      expect(formatFileSize(1073741824)).toBe('1 GB');
    });
  });

  describe('formatNumber', () => {
    test('应该正确格式化数字', () => {
      expect(formatNumber(1000)).toBe('1,000');
      expect(formatNumber(1234567)).toBe('1,234,567');
      expect(formatNumber(123)).toBe('123');
    });
  });

  describe('formatPercentage', () => {
    test('应该正确格式化百分比', () => {
      expect(formatPercentage(0.5)).toBe('50.0%');
      expect(formatPercentage(0.8567)).toBe('85.7%');
      expect(formatPercentage(0.8567, 2)).toBe('85.67%');
      expect(formatPercentage(1)).toBe('100.0%');
    });
  });

  describe('formatLineCount', () => {
    test('应该正确格式化代码行数', () => {
      expect(formatLineCount(100)).toBe('100 行');
      expect(formatLineCount(1500)).toBe('1.5K 行');
      expect(formatLineCount(1000000)).toBe('1.0M 行');
      expect(formatLineCount(2500000)).toBe('2.5M 行');
    });
  });

  describe('formatComplexityLevel', () => {
    test('应该正确判断复杂度等级', () => {
      expect(formatComplexityLevel(3)).toEqual({ level: '简单', color: 'green' });
      expect(formatComplexityLevel(8)).toEqual({ level: '中等', color: 'orange' });
      expect(formatComplexityLevel(15)).toEqual({ level: '复杂', color: 'red' });
      expect(formatComplexityLevel(25)).toEqual({ level: '极复杂', color: 'darkred' });
    });
  });

  describe('formatSecurityLevel', () => {
    test('应该正确格式化安全等级', () => {
      expect(formatSecurityLevel('low')).toEqual({ text: '低风险', color: 'green' });
      expect(formatSecurityLevel('medium')).toEqual({ text: '中风险', color: 'orange' });
      expect(formatSecurityLevel('high')).toEqual({ text: '高风险', color: 'red' });
      expect(formatSecurityLevel('critical')).toEqual({ text: '严重', color: 'darkred' });
    });
  });

  describe('truncateText', () => {
    test('应该正确截断长文本', () => {
      const longText = 'This is a very long text that should be truncated';
      expect(truncateText(longText, 20)).toBe('This is a very long ...');
    });

    test('应该保持短文本不变', () => {
      const shortText = 'Short text';
      expect(truncateText(shortText, 20)).toBe('Short text');
    });

    test('应该使用默认长度', () => {
      const text = 'a'.repeat(150);
      const result = truncateText(text);
      expect(result).toBe('a'.repeat(100) + '...');
    });
  });

  describe('highlightSearchTerm', () => {
    test('应该正确高亮搜索关键词', () => {
      const text = 'This is a test text';
      const searchTerm = 'test';
      expect(highlightSearchTerm(text, searchTerm)).toBe('This is a <mark>test</mark> text');
    });

    test('应该忽略大小写', () => {
      const text = 'This is a Test text';
      const searchTerm = 'test';
      expect(highlightSearchTerm(text, searchTerm)).toBe('This is a <mark>Test</mark> text');
    });

    test('应该处理空搜索词', () => {
      const text = 'This is a test text';
      expect(highlightSearchTerm(text, '')).toBe('This is a test text');
    });

    test('应该高亮多个匹配项', () => {
      const text = 'test this test';
      const searchTerm = 'test';
      expect(highlightSearchTerm(text, searchTerm)).toBe('<mark>test</mark> this <mark>test</mark>');
    });
  });
});