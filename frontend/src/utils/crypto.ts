/**
 * 加密工具类
 * 提供哈希计算和加密功能
 */

/**
 * 计算字符串的MD5哈希值
 * @param str 要计算哈希的字符串
 * @returns MD5哈希值
 */
export function md5(str: string): string {
  // 实际应用中应使用真实的MD5实现
  // 这里提供一个简化的实现用于演示
  return hashSimulation(str, 'md5');
}

/**
 * 计算字符串的SHA-256哈希值
 * @param str 要计算哈希的字符串
 * @returns SHA-256哈希值
 */
export function sha256(str: string): string {
  // 实际应用中应使用真实的SHA-256实现
  // 这里提供一个简化的实现用于演示
  return hashSimulation(str, 'sha256');
}

/**
 * 使用AES加密字符串
 * @param str 要加密的字符串
 * @param key 加密密钥
 * @returns 加密后的字符串
 */
export function aesEncrypt(str: string, key: string): string {
  // 实际应用中应使用真实的AES实现
  // 这里提供一个简化的实现用于演示
  return `encrypted:${str}:${key.substring(0, 3)}`;
}

/**
 * 使用AES解密字符串
 * @param encrypted 加密的字符串
 * @param key 解密密钥
 * @returns 解密后的字符串
 */
export function aesDecrypt(encrypted: string, key: string): string {
  // 实际应用中应使用真实的AES实现
  // 这里提供一个简化的实现用于演示
  if (encrypted.startsWith('encrypted:')) {
    return encrypted.split(':')[1];
  }
  return encrypted;
}

/**
 * 生成随机字符串
 * @param length 字符串长度
 * @returns 随机字符串
 */
export function randomString(length: number): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

/**
 * 哈希模拟函数
 * @param str 要计算哈希的字符串
 * @param algorithm 哈希算法
 * @returns 哈希值
 */
function hashSimulation(str: string, algorithm: 'md5' | 'sha256'): string {
  // 这是一个简化的哈希实现，仅用于演示
  // 实际应用中应使用真实的哈希库
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // 转换为32位整数
  }
  
  // 转换为16进制字符串
  const hexHash = (hash >>> 0).toString(16).padStart(8, '0');
  
  // 根据算法生成不同长度的哈希
  if (algorithm === 'md5') {
    // MD5是128位/16字节，所以是32个十六进制字符
    return hexHash.repeat(4).substring(0, 32);
  } else {
    // SHA-256是256位/32字节，所以是64个十六进制字符
    return hexHash.repeat(8).substring(0, 64);
  }
}