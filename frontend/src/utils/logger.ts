/**
 * 统一日志工具类
 * 用于替代直接使用console.log，提供更一致的日志格式和级别控制
 */

// 日志级别枚举
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4, // 用于完全禁用日志
}

// 当前环境配置
const isProduction = import.meta.env.MODE === 'production';

// 默认日志级别：生产环境使用WARN，开发环境使用DEBUG
let currentLogLevel = isProduction ? LogLevel.WARN : LogLevel.DEBUG;

// 日志前缀格式化
const formatPrefix = (level: string): string => {
  const timestamp = new Date().toISOString();
  return `[${timestamp}] [${level}]`;
};

/**
 * 日志工具类
 */
export class Logger {
  private context: string;

  /**
   * 创建日志实例
   * @param context 日志上下文名称，通常是组件或服务名
   */
  constructor(context: string) {
    this.context = context;
  }

  /**
   * 设置全局日志级别
   * @param level 日志级别
   */
  static setLogLevel(level: LogLevel): void {
    currentLogLevel = level;
  }

  /**
   * 获取当前日志级别
   */
  static getLogLevel(): LogLevel {
    return currentLogLevel;
  }

  /**
   * 调试日志
   * @param message 日志消息
   * @param args 附加参数
   */
  debug(message: string, ...args: any[]): void {
    if (currentLogLevel <= LogLevel.DEBUG) {
      console.debug(`${formatPrefix('DEBUG')} [${this.context}]`, message, ...args);
    }
  }

  /**
   * 信息日志
   * @param message 日志消息
   * @param args 附加参数
   */
  info(message: string, ...args: any[]): void {
    if (currentLogLevel <= LogLevel.INFO) {
      console.info(`${formatPrefix('INFO')} [${this.context}]`, message, ...args);
    }
  }

  /**
   * 警告日志
   * @param message 日志消息
   * @param args 附加参数
   */
  warn(message: string, ...args: any[]): void {
    if (currentLogLevel <= LogLevel.WARN) {
      console.warn(`${formatPrefix('WARN')} [${this.context}]`, message, ...args);
    }
  }

  /**
   * 错误日志
   * @param message 日志消息
   * @param args 附加参数
   */
  error(message: string, ...args: any[]): void {
    if (currentLogLevel <= LogLevel.ERROR) {
      console.error(`${formatPrefix('ERROR')} [${this.context}]`, message, ...args);
    }
  }

  /**
   * 记录性能计时开始
   * @param label 计时标签
   */
  time(label: string): void {
    if (currentLogLevel <= LogLevel.DEBUG) {
      console.time(`[${this.context}] ${label}`);
    }
  }

  /**
   * 记录性能计时结束并输出
   * @param label 计时标签
   */
  timeEnd(label: string): void {
    if (currentLogLevel <= LogLevel.DEBUG) {
      console.timeEnd(`[${this.context}] ${label}`);
    }
  }
}

// 创建默认日志实例
export const defaultLogger = new Logger('App');

// 导出单例方法，方便直接使用
export const debug = defaultLogger.debug.bind(defaultLogger);
export const info = defaultLogger.info.bind(defaultLogger);
export const warn = defaultLogger.warn.bind(defaultLogger);
export const error = defaultLogger.error.bind(defaultLogger);
