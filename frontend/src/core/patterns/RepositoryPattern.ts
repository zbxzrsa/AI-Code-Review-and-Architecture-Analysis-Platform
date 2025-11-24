/**
 * 仓储模式实现
 * 用于数据访问层抽象，提高代码可测试性和可维护性
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { addSecurityHeaders } from '../../utils/securityUtils';

// 基础仓储接口
export interface IRepository<T, ID> {
  findAll(): Promise<T[]>;
  findById(id: ID): Promise<T | null>;
  create(entity: Omit<T, 'id'>): Promise<T>;
  update(id: ID, entity: Partial<T>): Promise<T>;
  delete(id: ID): Promise<boolean>;
}

// HTTP仓储基类
export abstract class HttpRepository<T, ID> implements IRepository<T, ID> {
  protected client: AxiosInstance;
  protected baseUrl: string;

  constructor(baseUrl: string, config?: AxiosRequestConfig) {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      ...config,
    });

    // 请求拦截器：添加安全头部
    this.client.interceptors.request.use((config) => {
      const headers = addSecurityHeaders();
      if ((config.headers as any)?.set) {
        Object.entries(headers).forEach(([k, v]) => (config.headers as any).set(k, v as any));
      } else {
        config.headers = { ...(config.headers as any), ...headers } as any;
      }
      return config;
    });
  }

  async findAll(): Promise<T[]> {
    const response = await this.client.get<T[]>('');
    return response.data;
  }

  async findById(id: ID): Promise<T | null> {
    try {
      const response = await this.client.get<T>(`/${id}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async create(entity: Omit<T, 'id'>): Promise<T> {
    const response = await this.client.post<T>('', entity);
    return response.data;
  }

  async update(id: ID, entity: Partial<T>): Promise<T> {
    const response = await this.client.put<T>(`/${id}`, entity);
    return response.data;
  }

  async delete(id: ID): Promise<boolean> {
    await this.client.delete(`/${id}`);
    return true;
  }

  // 自定义查询方法
  protected async query<R>(path: string, params?: any): Promise<R> {
    const response = await this.client.get<R>(path, { params });
    return response.data;
  }

  // 自定义命令方法
  protected async command<R>(path: string, data?: any): Promise<R> {
    const response = await this.client.post<R>(path, data);
    return response.data;
  }
}

// 本地存储仓储基类
export abstract class LocalStorageRepository<T extends { id: ID }, ID> implements IRepository<T, ID> {
  protected storageKey: string;

  constructor(storageKey: string) {
    this.storageKey = storageKey;
  }

  protected getItems(): T[] {
    const data = localStorage.getItem(this.storageKey);
    return data ? JSON.parse(data) : [];
  }

  protected saveItems(items: T[]): void {
    localStorage.setItem(this.storageKey, JSON.stringify(items));
  }

  async findAll(): Promise<T[]> {
    return this.getItems();
  }

  async findById(id: ID): Promise<T | null> {
    const items = this.getItems();
    const item = items.find(item => item.id === id);
    return item || null;
  }

  async create(entity: Omit<T, 'id'>): Promise<T> {
    const items = this.getItems();
    const newId = this.generateId();
    const newItem = { ...entity, id: newId } as T;
    
    items.push(newItem);
    this.saveItems(items);
    
    return newItem;
  }

  async update(id: ID, entity: Partial<T>): Promise<T> {
    const items = this.getItems();
    const index = items.findIndex(item => item.id === id);
    
    if (index === -1) {
      throw new Error(`Entity with id ${id} not found`);
    }
    
    const updatedItem = { ...items[index], ...entity };
    items[index] = updatedItem;
    this.saveItems(items);
    
    return updatedItem;
  }

  async delete(id: ID): Promise<boolean> {
    const items = this.getItems();
    const filteredItems = items.filter(item => item.id !== id);
    
    if (filteredItems.length === items.length) {
      return false;
    }
    
    this.saveItems(filteredItems);
    return true;
  }

  protected abstract generateId(): ID;
}

// 内存仓储基类（用于测试）
export abstract class InMemoryRepository<T extends { id: ID }, ID> implements IRepository<T, ID> {
  protected items: T[] = [];

  async findAll(): Promise<T[]> {
    return [...this.items];
  }

  async findById(id: ID): Promise<T | null> {
    const item = this.items.find(item => item.id === id);
    return item ? { ...item } : null;
  }

  async create(entity: Omit<T, 'id'>): Promise<T> {
    const newId = this.generateId();
    const newItem = { ...entity, id: newId } as T;
    
    this.items.push(newItem);
    return { ...newItem };
  }

  async update(id: ID, entity: Partial<T>): Promise<T> {
    const index = this.items.findIndex(item => item.id === id);
    
    if (index === -1) {
      throw new Error(`Entity with id ${id} not found`);
    }
    
    const updatedItem = { ...this.items[index], ...entity };
    this.items[index] = updatedItem;
    
    return { ...updatedItem };
  }

  async delete(id: ID): Promise<boolean> {
    const initialLength = this.items.length;
    this.items = this.items.filter(item => item.id !== id);
    
    return this.items.length !== initialLength;
  }

  protected abstract generateId(): ID;
}

// 使用示例
/*
// 项目仓储实现
interface Project {
  id: number;
  name: string;
  description: string;
  createdAt: string;
}

// HTTP实现
class ProjectHttpRepository extends HttpRepository<Project, number> {
  constructor() {
    super('/api/projects');
  }
  
  // 自定义方法
  async findByName(name: string): Promise<Project[]> {
    return this.query<Project[]>('/search', { name });
  }
}

// 本地存储实现
class ProjectLocalStorageRepository extends LocalStorageRepository<Project, number> {
  constructor() {
    super('projects');
  }
  
  protected generateId(): number {
    const items = this.getItems();
    return items.length > 0 
      ? Math.max(...items.map(item => item.id)) + 1 
      : 1;
  }
}
*/
