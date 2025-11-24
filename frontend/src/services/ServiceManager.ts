import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';

export interface ServiceStatus {
  backend: 'stopped' | 'starting' | 'running' | 'error';
  frontend: 'stopped' | 'starting' | 'running' | 'error';
  docker: 'stopped' | 'starting' | 'running' | 'error';
}

export class ServiceManager extends EventEmitter {
  private backendProcess: ChildProcess | null = null;
  private frontendProcess: ChildProcess | null = null;
  private dockerProcess: ChildProcess | null = null;
  private projectRoot: string;
  
  constructor() {
    super();
    this.projectRoot = process.cwd();
  }
  
  /**
   * 检查 Docker 是否可用
   */
  async checkDocker(): Promise<boolean> {
    return new Promise((resolve) => {
      const child = spawn('docker', ['--version']);
      
      child.on('close', (code) => {
        resolve(code === 0);
      });
      
      child.on('error', () => {
        resolve(false);
      });
    });
  }
  
  /**
   * 启动后端服务
   */
  async startBackend(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.backendProcess) {
        return resolve();
      }
      
      try {
        // 切换到后端目录
        const backendDir = `${this.projectRoot}/backend`;
        
        // 使用 Python 启动后端服务
        this.backendProcess = spawn('python', ['app.py'], {
          cwd: backendDir,
          stdio: 'pipe'
        });
        
        this.backendProcess.stdout?.on('data', (data) => {
          const message = data.toString().trim();
          if (message) {
            this.emit('backend-log', message);
          }
        });
        
        this.backendProcess.stderr?.on('data', (data) => {
          const message = data.toString().trim();
          if (message) {
            this.emit('backend-error', message);
          }
        });
        
        this.backendProcess.on('close', (code) => {
          this.backendProcess = null;
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`后端服务启动失败，退出码: ${code}`));
          }
        });
        
        this.backendProcess.on('error', (error) => {
          this.backendProcess = null;
          reject(error);
        });
        
        // 等待几秒钟让服务启动
        setTimeout(() => {
          this.emit('backend-status', 'running');
          resolve();
        }, 5000);
        
      } catch (error) {
        this.backendProcess = null;
        reject(error);
      }
    });
  }
  
  /**
   * 停止后端服务
   */
  async stopBackend(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.backendProcess) {
        return resolve();
      }
      
      try {
        this.backendProcess.kill('SIGTERM');
        
        this.backendProcess.on('close', () => {
          this.backendProcess = null;
          this.emit('backend-status', 'stopped');
          resolve();
        });
        
        // 如果 SIGTERM 不起作用，使用 SIGKILL
        setTimeout(() => {
          if (this.backendProcess) {
            this.backendProcess.kill('SIGKILL');
          }
        }, 3000);
        
      } catch (error) {
        this.backendProcess = null;
        reject(error);
      }
    });
  }
  
  /**
   * 启动前端服务
   */
  async startFrontend(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.frontendProcess) {
        return resolve();
      }
      
      try {
        // 切换到前端目录
        const frontendDir = `${this.projectRoot}/frontend`;
        
        // 使用 npm 启动前端服务
        this.frontendProcess = spawn('npm', ['start'], {
          cwd: frontendDir,
          stdio: 'pipe'
        });
        
        this.frontendProcess.stdout?.on('data', (data) => {
          const message = data.toString().trim();
          if (message) {
            this.emit('frontend-log', message);
          }
        });
        
        this.frontendProcess.stderr?.on('data', (data) => {
          const message = data.toString().trim();
          if (message) {
            this.emit('frontend-error', message);
          }
        });
        
        this.frontendProcess.on('close', (code) => {
          this.frontendProcess = null;
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`前端服务启动失败，退出码: ${code}`));
          }
        });
        
        this.frontendProcess.on('error', (error) => {
          this.frontendProcess = null;
          reject(error);
        });
        
        // 等待几秒钟让服务启动
        setTimeout(() => {
          this.emit('frontend-status', 'running');
          resolve();
        }, 5000);
        
      } catch (error) {
        this.frontendProcess = null;
        reject(error);
      }
    });
  }
  
  /**
   * 停止前端服务
   */
  async stopFrontend(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.frontendProcess) {
        return resolve();
      }
      
      try {
        this.frontendProcess.kill('SIGTERM');
        
        this.frontendProcess.on('close', () => {
          this.frontendProcess = null;
          this.emit('frontend-status', 'stopped');
          resolve();
        });
        
        // 如果 SIGTERM 不起作用，使用 SIGKILL
        setTimeout(() => {
          if (this.frontendProcess) {
            this.frontendProcess.kill('SIGKILL');
          }
        }, 3000);
        
      } catch (error) {
        this.frontendProcess = null;
        reject(error);
      }
    });
  }
  
  /**
   * 启动 Docker 服务
   */
  async startDocker(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.dockerProcess) {
        return resolve();
      }
      
      try {
        // 切换到项目根目录
        const rootDir = this.projectRoot;
        
        // 使用 docker-compose 启动服务
        this.dockerProcess = spawn('docker-compose', ['up', '-d'], {
          cwd: rootDir,
          stdio: 'pipe'
        });
        
        this.dockerProcess.stdout?.on('data', (data) => {
          const message = data.toString().trim();
          if (message) {
            this.emit('docker-log', message);
          }
        });
        
        this.dockerProcess.stderr?.on('data', (data) => {
          const message = data.toString().trim();
          if (message) {
            this.emit('docker-error', message);
          }
        });
        
        this.dockerProcess.on('close', (code) => {
          this.dockerProcess = null;
          if (code === 0) {
            this.emit('docker-status', 'running');
            resolve();
          } else {
            reject(new Error(`Docker 服务启动失败，退出码: ${code}`));
          }
        });
        
        this.dockerProcess.on('error', (error) => {
          this.dockerProcess = null;
          reject(error);
        });
        
        // 等待几秒钟让服务启动
        setTimeout(() => {
          this.emit('docker-status', 'running');
          resolve();
        }, 10000);
        
      } catch (error) {
        this.dockerProcess = null;
        reject(error);
      }
    });
  }
  
  /**
   * 停止 Docker 服务
   */
  async stopDocker(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.dockerProcess) {
        // 如果没有正在运行的 docker-compose 进程，尝试停止现有的容器
        this.stopExistingDockerContainers().then(resolve).catch(reject);
        return;
      }
      
      try {
        this.dockerProcess.kill('SIGTERM');
        
        this.dockerProcess.on('close', () => {
          this.dockerProcess = null;
          this.emit('docker-status', 'stopped');
          resolve();
        });
        
        // 如果 SIGTERM 不起作用，使用 SIGKILL
        setTimeout(() => {
          if (this.dockerProcess) {
            this.dockerProcess.kill('SIGKILL');
          }
        }, 3000);
        
      } catch (error) {
        this.dockerProcess = null;
        reject(error);
      }
    });
  }
  
  /**
   * 停止现有的 Docker 容器
   */
  private async stopExistingDockerContainers(): Promise<void> {
    return new Promise((resolve, reject) => {
      const child = spawn('docker-compose', ['down'], {
        cwd: this.projectRoot,
        stdio: 'pipe'
      });
      
      child.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`停止 Docker 容器失败，退出码: ${code}`));
        }
      });
      
      child.on('error', (error) => {
        reject(error);
      });
    });
  }
  
  /**
   * 获取所有服务的状态
   */
  async getServiceStatus(): Promise<ServiceStatus> {
    const status: ServiceStatus = {
      backend: 'stopped',
      frontend: 'stopped',
      docker: 'stopped'
    };
    
    // 检查后端服务状态
    if (this.backendProcess) {
      status.backend = 'running';
    }
    
    // 检查前端服务状态
    if (this.frontendProcess) {
      status.frontend = 'running';
    }
    
    // 检查 Docker 服务状态
    if (this.dockerProcess) {
      status.docker = 'running';
    }
    
    return status;
  }
  
  /**
   * 停止所有服务
   */
  async stopAll(): Promise<void> {
    await Promise.all([
      this.stopDocker(),
      this.stopFrontend(),
      this.stopBackend()
    ]);
  }
  
  /**
   * 销毁服务管理器
   */
  destroy(): void {
    this.stopAll();
    this.removeAllListeners();
  }
}