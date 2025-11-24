/**
 * 验证工具函数单元测试
 */
import {
  validateEmail,
  validatePassword,
  validateProjectName,
  validateCode,
  validateUrl,
  validatePhoneNumber,
  validateFileType,
  validateFileSize,
  validateRequired,
  validateNumberRange
} from '../validation';

describe('验证工具函数测试', () => {
  describe('validateEmail', () => {
    test('应该验证有效邮箱', () => {
      expect(validateEmail('test@example.com')).toBe(true);
      expect(validateEmail('user.name@domain.co.uk')).toBe(true);
      expect(validateEmail('test+tag@example.org')).toBe(true);
    });

    test('应该拒绝无效邮箱', () => {
      expect(validateEmail('invalid-email')).toBe(false);
      expect(validateEmail('test@')).toBe(false);
      expect(validateEmail('@example.com')).toBe(false);
      expect(validateEmail('test.example.com')).toBe(false);
    });
  });

  describe('validatePassword', () => {
    test('应该验证强密码', () => {
      const result = validatePassword('StrongPass123!');
      expect(result.isValid).toBe(true);
      expect(result.strength).toBe('强');
      expect(result.errors).toHaveLength(0);
    });

    test('应该识别中等强度密码', () => {
      const result = validatePassword('GoodPass123');
      expect(result.isValid).toBe(false);
      expect(result.strength).toBe('中');
      expect(result.errors).toContain('密码必须包含特殊字符');
    });

    test('应该识别弱密码', () => {
      const result = validatePassword('weak');
      expect(result.isValid).toBe(false);
      expect(result.strength).toBe('弱');
      expect(result.errors.length).toBeGreaterThan(0);
    });

    test('应该检查密码长度', () => {
      const result = validatePassword('Short1!');
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('密码长度至少8位');
    });
  });

  describe('validateProjectName', () => {
    test('应该验证有效项目名称', () => {
      expect(validateProjectName('MyProject')).toEqual({ isValid: true });
      expect(validateProjectName('项目名称')).toEqual({ isValid: true });
      expect(validateProjectName('project_123')).toEqual({ isValid: true });
      expect(validateProjectName('project-name')).toEqual({ isValid: true });
    });

    test('应该拒绝空项目名称', () => {
      expect(validateProjectName('')).toEqual({ 
        isValid: false, 
        error: '项目名称不能为空' 
      });
      expect(validateProjectName('   ')).toEqual({ 
        isValid: false, 
        error: '项目名称不能为空' 
      });
    });

    test('应该检查项目名称长度', () => {
      expect(validateProjectName('a')).toEqual({ 
        isValid: false, 
        error: '项目名称至少2个字符' 
      });
      
      const longName = 'a'.repeat(51);
      expect(validateProjectName(longName)).toEqual({ 
        isValid: false, 
        error: '项目名称不能超过50个字符' 
      });
    });

    test('应该拒绝包含特殊字符的项目名称', () => {
      expect(validateProjectName('project@name')).toEqual({ 
        isValid: false, 
        error: '项目名称只能包含字母、数字、中文、下划线和连字符' 
      });
    });
  });

  describe('validateCode', () => {
    test('应该验证有效代码', () => {
      const code = 'function test() { return true; }';
      expect(validateCode(code)).toEqual({ isValid: true });
    });

    test('应该拒绝空代码', () => {
      expect(validateCode('')).toEqual({ 
        isValid: false, 
        error: '代码内容不能为空' 
      });
      expect(validateCode('   ')).toEqual({ 
        isValid: false, 
        error: '代码内容不能为空' 
      });
    });

    test('应该检查代码长度', () => {
      const longCode = 'a'.repeat(100001);
      expect(validateCode(longCode)).toEqual({ 
        isValid: false, 
        error: '代码内容不能超过100KB' 
      });
    });

    test('应该检测潜在恶意代码', () => {
      expect(validateCode('eval("malicious code")')).toEqual({ 
        isValid: false, 
        error: '代码包含潜在的安全风险' 
      });
      
      expect(validateCode('system("rm -rf /")')).toEqual({ 
        isValid: false, 
        error: '代码包含潜在的安全风险' 
      });
    });
  });

  describe('validateUrl', () => {
    test('应该验证有效URL', () => {
      expect(validateUrl('https://example.com')).toBe(true);
      expect(validateUrl('http://localhost:3000')).toBe(true);
      expect(validateUrl('ftp://files.example.com')).toBe(true);
    });

    test('应该拒绝无效URL', () => {
      expect(validateUrl('not-a-url')).toBe(false);
      expect(validateUrl('http://')).toBe(false);
      expect(validateUrl('')).toBe(false);
    });
  });

  describe('validatePhoneNumber', () => {
    test('应该验证有效中国手机号', () => {
      expect(validatePhoneNumber('13812345678')).toBe(true);
      expect(validatePhoneNumber('15987654321')).toBe(true);
      expect(validatePhoneNumber('18612345678')).toBe(true);
    });

    test('应该拒绝无效手机号', () => {
      expect(validatePhoneNumber('12812345678')).toBe(false); // 不以1开头
      expect(validatePhoneNumber('1381234567')).toBe(false);  // 长度不够
      expect(validatePhoneNumber('138123456789')).toBe(false); // 长度过长
      expect(validatePhoneNumber('abcdefghijk')).toBe(false);  // 包含字母
    });
  });

  describe('validateFileType', () => {
    test('应该验证允许的文件类型', () => {
      const file = new File([''], 'test.jpg', { type: 'image/jpeg' });
      const allowedTypes = ['image/jpeg', 'image/png'];
      
      expect(validateFileType(file, allowedTypes)).toEqual({ isValid: true });
    });

    test('应该拒绝不允许的文件类型', () => {
      const file = new File([''], 'test.txt', { type: 'text/plain' });
      const allowedTypes = ['image/jpeg', 'image/png'];
      
      expect(validateFileType(file, allowedTypes)).toEqual({ 
        isValid: false, 
        error: '不支持的文件类型，仅支持: image/jpeg, image/png' 
      });
    });
  });

  describe('validateFileSize', () => {
    test('应该验证允许的文件大小', () => {
      const file = new File(['a'.repeat(1024)], 'test.txt', { type: 'text/plain' });
      
      expect(validateFileSize(file, 1)).toEqual({ isValid: true });
    });

    test('应该拒绝超大文件', () => {
      const file = new File(['a'.repeat(2 * 1024 * 1024)], 'test.txt', { type: 'text/plain' });
      
      expect(validateFileSize(file, 1)).toEqual({ 
        isValid: false, 
        error: '文件大小不能超过 1MB' 
      });
    });
  });

  describe('validateRequired', () => {
    test('应该验证非空值', () => {
      expect(validateRequired('value', '字段')).toEqual({ isValid: true });
      expect(validateRequired(0, '数字')).toEqual({ isValid: true });
      expect(validateRequired(false, '布尔值')).toEqual({ isValid: true });
    });

    test('应该拒绝空值', () => {
      expect(validateRequired('', '字段')).toEqual({ 
        isValid: false, 
        error: '字段不能为空' 
      });
      expect(validateRequired(null, '字段')).toEqual({ 
        isValid: false, 
        error: '字段不能为空' 
      });
      expect(validateRequired(undefined, '字段')).toEqual({ 
        isValid: false, 
        error: '字段不能为空' 
      });
    });
  });

  describe('validateNumberRange', () => {
    test('应该验证范围内的数字', () => {
      expect(validateNumberRange(5, 1, 10, '数值')).toEqual({ isValid: true });
      expect(validateNumberRange(1, 1, 10, '数值')).toEqual({ isValid: true });
      expect(validateNumberRange(10, 1, 10, '数值')).toEqual({ isValid: true });
    });

    test('应该拒绝非数字值', () => {
      expect(validateNumberRange(NaN, 1, 10, '数值')).toEqual({ 
        isValid: false, 
        error: '数值必须是数字' 
      });
    });

    test('应该拒绝超出范围的数字', () => {
      expect(validateNumberRange(0, 1, 10, '数值')).toEqual({ 
        isValid: false, 
        error: '数值必须在 1 到 10 之间' 
      });
      expect(validateNumberRange(11, 1, 10, '数值')).toEqual({ 
        isValid: false, 
        error: '数值必须在 1 到 10 之间' 
      });
    });
  });
});