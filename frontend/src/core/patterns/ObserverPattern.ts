/**
 * 观察者模式实现
 * 用于组件间通信和状态管理
 */

export type Observer<T> = (data: T) => void;

export class Observable<T> {
  private observers: Observer<T>[] = [];

  /**
   * 订阅观察者
   * @param observer 观察者函数
   * @returns 取消订阅的函数
   */
  subscribe(observer: Observer<T>): () => void {
    this.observers.push(observer);
    
    // 返回取消订阅函数
    return () => {
      this.observers = this.observers.filter(obs => obs !== observer);
    };
  }

  /**
   * 通知所有观察者
   * @param data 通知数据
   */
  notify(data: T): void {
    this.observers.forEach(observer => observer(data));
  }

  /**
   * 获取观察者数量
   */
  get observerCount(): number {
    return this.observers.length;
  }

  /**
   * 清除所有观察者
   */
  clear(): void {
    this.observers = [];
  }
}

/**
 * 创建事件总线
 * 用于跨组件通信
 */
export class EventBus {
  private static instance: EventBus;
  private events: Map<string, Observable<any>> = new Map();

  private constructor() {}

  /**
   * 获取EventBus单例
   */
  static getInstance(): EventBus {
    if (!EventBus.instance) {
      EventBus.instance = new EventBus();
    }
    return EventBus.instance;
  }

  /**
   * 订阅事件
   * @param eventName 事件名称
   * @param callback 回调函数
   * @returns 取消订阅的函数
   */
  on<T>(eventName: string, callback: (data: T) => void): () => void {
    if (!this.events.has(eventName)) {
      this.events.set(eventName, new Observable<T>());
    }
    
    const observable = this.events.get(eventName) as Observable<T>;
    return observable.subscribe(callback);
  }

  /**
   * 发布事件
   * @param eventName 事件名称
   * @param data 事件数据
   */
  emit<T>(eventName: string, data: T): void {
    if (!this.events.has(eventName)) {
      return;
    }
    
    const observable = this.events.get(eventName) as Observable<T>;
    observable.notify(data);
  }

  /**
   * 移除特定事件的所有监听器
   * @param eventName 事件名称
   */
  off(eventName: string): void {
    if (this.events.has(eventName)) {
      const observable = this.events.get(eventName);
      observable?.clear();
      this.events.delete(eventName);
    }
  }

  /**
   * 清除所有事件
   */
  clear(): void {
    this.events.forEach(observable => observable.clear());
    this.events.clear();
  }
}

// 导出单例实例
export const eventBus = EventBus.getInstance();

// 使用示例
/*
// 在组件A中
import { eventBus } from '../core/patterns/ObserverPattern';

// 订阅事件
const unsubscribe = eventBus.on<string>('message', (data) => {
  console.log('Received message:', data);
});

// 在组件B中
import { eventBus } from '../core/patterns/ObserverPattern';

// 发布事件
eventBus.emit('message', 'Hello from component B');

// 取消订阅
unsubscribe();
*/