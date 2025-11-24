/**
 * 安全工具函数集合
 * 提供输入验证、XSS防护和CSRF防护等功能
 */

/**
 * 对HTML内容进行转义，防止XSS攻击
 * @param html 需要转义的HTML字符串
 * @returns 转义后的安全字符串
 */
export function escapeHtml(html: string): string {
  if (!html) return '';
  
  return html
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * 验证输入是否为有效的电子邮件地址
 * @param email 需要验证的电子邮件地址
 * @returns 是否为有效的电子邮件地址
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return emailRegex.test(email);
}

/**
 * 验证输入是否为有效的URL
 * @param url 需要验证的URL
 * @returns 是否为有效的URL
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * 获取CSRF令牌
 * @returns CSRF令牌
 */
export function getCsrfToken(): string {
  // 从cookie或meta标签中获取CSRF令牌
  const metaTag = document.querySelector('meta[name="csrf-token"]');
  if (metaTag) {
    return metaTag.getAttribute('content') || '';
  }
  return '';
}

/**
 * 为API请求添加安全头部
 * @param headers 原始请求头部
 * @returns 添加了安全头部的请求头部
 */
export function addSecurityHeaders(headers: Record<string, string> = {}): Record<string, string> {
  const csrfToken = getCsrfToken();
  
  return {
    ...headers,
    'X-CSRF-Token': csrfToken,
    'Content-Security-Policy': "default-src 'self'",
  };
}

/**
 * 安全地解析JSON，防止JSON注入攻击
 * @param jsonString 需要解析的JSON字符串
 * @returns 解析后的对象，解析失败则返回null
 */
export function safeJsonParse(jsonString: string): any {
  try {
    return JSON.parse(jsonString);
  } catch (e) {
    console.error('JSON解析失败:', e);
    return null;
  }
}

/**
 * 对用户输入进行清理，移除潜在的危险字符
 * @param input 用户输入
 * @returns 清理后的安全字符串
 */
export function sanitizeInput(input: string): string {
  if (!input) return '';
  
  // 移除可能导致XSS的脚本标签和事件处理程序
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/on\w+="[^"]*"/g, '')
    .replace(/on\w+='[^']*'/g, '')
    .replace(/on\w+=\w+/g, '');
}

/**
 * 生成随机ID，用于DOM元素等
 * @param prefix ID前缀
 * @returns 随机生成的ID
 */
export function generateSafeId(prefix: string = 'id'): string {
  return `${prefix}-${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * 检查密码强度
 * @param password 密码
 * @returns 密码强度评分（0-100）
 */
export function checkPasswordStrength(password: string): number {
  if (!password) return 0;
  
  let score = 0;
  
  // 长度检查
  if (password.length >= 8) score += 20;
  if (password.length >= 12) score += 10;
  
  // 复杂性检查
  if (/[a-z]/.test(password)) score += 10;
  if (/[A-Z]/.test(password)) score += 10;
  if (/[0-9]/.test(password)) score += 10;
  if (/[^a-zA-Z0-9]/.test(password)) score += 10;
  
  // 多样性检查
  const uniqueChars = new Set(password.split('')).size;
  score += Math.min(20, uniqueChars * 2);
  
  // 常见模式检查
  if (/123|abc|qwerty|password|admin/i.test(password)) score -= 20;
  
  return Math.max(0, Math.min(100, score));
}

/**
 * 安全地存储敏感数据到localStorage
 * @param key 存储键
 * @param value 存储值
 */
export function secureLocalStorage(key: string, value: any): void {
  try {
    // 简单加密，实际项目中应使用更强的加密方法
    const encodedValue = btoa(JSON.stringify(value));
    localStorage.setItem(key, encodedValue);
  } catch (e) {
    console.error('本地存储失败:', e);
  }
}

/**
 * 从localStorage安全地获取敏感数据
 * @param key 存储键
 * @returns 存储值，获取失败则返回null
 */
export function getSecureLocalStorage(key: string): any {
  try {
    const encodedValue = localStorage.getItem(key);
    if (!encodedValue) return null;
    
    // 解密
    return JSON.parse(atob(encodedValue));
  } catch (e) {
    console.error('从本地存储获取数据失败:', e);
    return null;
  }
}