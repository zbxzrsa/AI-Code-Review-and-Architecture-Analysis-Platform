/**
 * 验证工具函数
 * 提供表单验证、数据验证等功能
 */

// 邮箱验证
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// 密码强度验证
export const validatePassword = (password: string): { isValid: boolean; strength: string; errors: string[] } => {
  const errors: string[] = [];
  let strength = '弱';

  if (password.length < 8) {
    errors.push('密码长度至少8位');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('密码必须包含小写字母');
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('密码必须包含大写字母');
  }

  if (!/\d/.test(password)) {
    errors.push('密码必须包含数字');
  }

  if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    errors.push('密码必须包含特殊字符');
  }

  // 计算强度
  const hasLower = /[a-z]/.test(password);
  const hasUpper = /[A-Z]/.test(password);
  const hasNumber = /\d/.test(password);
  const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(password);
  const isLongEnough = password.length >= 8;

  const strengthScore = [hasLower, hasUpper, hasNumber, hasSpecial, isLongEnough].filter(Boolean).length;

  if (strengthScore >= 4) {
    strength = '强';
  } else if (strengthScore >= 3) {
    strength = '中';
  }

  return {
    isValid: errors.length === 0,
    strength,
    errors
  };
};

// 项目名称验证
export const validateProjectName = (name: string): { isValid: boolean; error?: string } => {
  if (!name || name.trim().length === 0) {
    return { isValid: false, error: '项目名称不能为空' };
  }

  if (name.length < 2) {
    return { isValid: false, error: '项目名称至少2个字符' };
  }

  if (name.length > 50) {
    return { isValid: false, error: '项目名称不能超过50个字符' };
  }

  if (!/^[a-zA-Z0-9\u4e00-\u9fa5_-]+$/.test(name)) {
    return { isValid: false, error: '项目名称只能包含字母、数字、中文、下划线和连字符' };
  }

  return { isValid: true };
};

// 代码内容验证
export const validateCode = (code: string): { isValid: boolean; error?: string } => {
  if (!code || code.trim().length === 0) {
    return { isValid: false, error: '代码内容不能为空' };
  }

  if (code.length > 100000) {
    return { isValid: false, error: '代码内容不能超过100KB' };
  }

  // 检查是否包含潜在的恶意代码
  const dangerousPatterns = [
    /eval\s*\(/,
    /exec\s*\(/,
    /system\s*\(/,
    /shell_exec\s*\(/,
    /passthru\s*\(/,
    /proc_open\s*\(/,
    /popen\s*\(/,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(code)) {
      return { isValid: false, error: '代码包含潜在的安全风险' };
    }
  }

  return { isValid: true };
};

// URL验证
export const validateUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

// 手机号验证（中国）
export const validatePhoneNumber = (phone: string): boolean => {
  const phoneRegex = /^1[3-9]\d{9}$/;
  return phoneRegex.test(phone);
};

// 文件类型验证
export const validateFileType = (file: File, allowedTypes: string[]): { isValid: boolean; error?: string } => {
  if (!allowedTypes.includes(file.type)) {
    return { 
      isValid: false, 
      error: `不支持的文件类型，仅支持: ${allowedTypes.join(', ')}` 
    };
  }
  return { isValid: true };
};

// 文件大小验证
export const validateFileSize = (file: File, maxSizeInMB: number): { isValid: boolean; error?: string } => {
  const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
  if (file.size > maxSizeInBytes) {
    return { 
      isValid: false, 
      error: `文件大小不能超过 ${maxSizeInMB}MB` 
    };
  }
  return { isValid: true };
};

// 通用必填字段验证
export const validateRequired = (value: any, fieldName: string): { isValid: boolean; error?: string } => {
  if (value === null || value === undefined || value === '') {
    return { isValid: false, error: `${fieldName}不能为空` };
  }
  return { isValid: true };
};

// 数字范围验证
export const validateNumberRange = (
  value: number, 
  min: number, 
  max: number, 
  fieldName: string
): { isValid: boolean; error?: string } => {
  if (isNaN(value)) {
    return { isValid: false, error: `${fieldName}必须是数字` };
  }
  
  if (value < min || value > max) {
    return { isValid: false, error: `${fieldName}必须在 ${min} 到 ${max} 之间` };
  }
  
  return { isValid: true };
};