#!/usr/bin/env node

/**
 * One-Command Runner for AI Code Review Platform
 * Cross-platform Docker orchestration with health checks
 */

import { spawn, exec } from 'child_process';
import { createInterface } from 'readline';
import { existsSync } from 'fs';
import { join } from 'path';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import fetch from 'node-fetch';

// Configuration
const CONFIG = {
  services: {
    backend: { port: 8000, healthPath: '/health' },
    frontend: { port: 3000, healthPath: '/' },
    postgres: { port: 5432, healthPath: null },
    redis: { port: 6379, healthPath: null },
    neo4j: { port: 7474, healthPath: '/' },
    prometheus: { port: 9090, healthPath: '/-/healthy' },
  },
  timeouts: {
    build: 300000, // 5 minutes
    health: 60000,  // 1 minute
    startup: 120000, // 2 minutes
  },
  retries: {
    health: 30,
    build: 3,
  }
};

// Utility functions
function log(message, type = 'info') {
  const timestamp = new Date().toISOString();
  const prefix = `[${timestamp}]`;
  
  switch (type) {
    case 'success':
      console.log(chalk.green(`${prefix} ✓ ${message}`));
      break;
    case 'error':
      console.log(chalk.red(`${prefix} ✗ ${message}`));
      break;
    case 'warning':
      console.log(chalk.yellow(`${prefix} ⚠ ${message}`));
      break;
    case 'info':
      console.log(chalk.blue(`${prefix} ℹ ${message}`));
      break;
    default:
      console.log(`${prefix} ${message}`);
  }
}

function execCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    log(`Executing: ${command} ${args.join(' ')}`, 'info');
    
    const child = spawn(command, args, {
      stdio: 'inherit',
      shell: true,
      ...options
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve(code);
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });

    child.on('error', (error) => {
      reject(error);
    });
  });
}

async function checkPort(port) {
  try {
    const response = await fetch(`http://localhost:${port}`, { 
      timeout: 5000,
      signal: AbortSignal.timeout(5000)
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function checkHealth(service) {
  const { port, healthPath } = CONFIG.services[service];
  if (!healthPath) {
    // For services without HTTP health checks, just check port
    return await checkPort(port);
  }

  try {
    const response = await fetch(`http://localhost:${port}${healthPath}`, {
      timeout: 5000,
      signal: AbortSignal.timeout(5000)
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForHealth(service, maxRetries = CONFIG.retries.health) {
  const spinner = ora(`Waiting for ${service} to be healthy...`).start();
  
  for (let i = 0; i < maxRetries; i++) {
    const isHealthy = await checkHealth(service);
    
    if (isHealthy) {
      spinner.succeed(`${service} is healthy`);
      return true;
    }
    
    spinner.text = `Waiting for ${service} to be healthy... (${i + 1}/${maxRetries})`;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  spinner.fail(`${service} failed to become healthy`);
  return false;
}

async function checkDockerInstallation() {
  try {
    await execCommand('docker', ['--version']);
    await execCommand('docker', ['compose', 'version']);
    return true;
  } catch {
    return false;
  }
}

async function checkPortConflicts() {
  log('Checking for port conflicts...', 'info');
  const conflicts = [];
  
  for (const [service, config] of Object.entries(CONFIG.services)) {
    const isPortInUse = await checkPort(config.port);
    if (isPortInUse) {
      conflicts.push({ service, port: config.port });
    }
  }
  
  if (conflicts.length > 0) {
    log('Port conflicts detected:', 'error');
    conflicts.forEach(({ service, port }) => {
      console.log(chalk.red(`  - ${service}: port ${port} is already in use`));
    });
    
    console.log(chalk.yellow('\nSolutions:'));
    console.log('1. Stop the conflicting services');
    console.log('2. Change the ports in docker-compose.yml');
    console.log('3. Use different PROFILE environment variable');
    
    return false;
  }
  
  log('No port conflicts detected', 'success');
  return true;
}

async function buildImages() {
  const spinner = ora('Building Docker images...').start();
  
  try {
    await execCommand('docker', ['compose', 'build'], {
      timeout: CONFIG.timeouts.build
    });
    spinner.succeed('Docker images built successfully');
    return true;
  } catch (error) {
    spinner.fail('Failed to build Docker images');
    log(`Build error: ${error.message}`, 'error');
    return false;
  }
}

async function startServices(profile = 'dev') {
  const spinner = ora('Starting services...').start();
  
  try {
    const composeArgs = ['compose', 'up', '-d'];
    
    // Add profile if specified
    if (profile && profile !== 'dev') {
      composeArgs.push('--profile', profile);
    }
    
    // Set environment variables
    const env = { ...process.env, PROFILE: profile };
    
    await execCommand('docker', composeArgs, { env });
    spinner.succeed('Services started');
    return true;
  } catch (error) {
    spinner.fail('Failed to start services');
    log(`Start error: ${error.message}`, 'error');
    return false;
  }
}

async function stopServices() {
  const spinner = ora('Stopping services...').start();
  
  try {
    await execCommand('docker', ['compose', 'down', '-v']);
    spinner.succeed('Services stopped');
    return true;
  } catch (error) {
    spinner.fail('Failed to stop services');
    log(`Stop error: ${error.message}`, 'error');
    return false;
  }
}

async function showLogs() {
  log('Showing service logs (Ctrl+C to exit)...', 'info');
  try {
    await execCommand('docker', ['compose', 'logs', '-f']);
  } catch (error) {
    log(`Failed to show logs: ${error.message}`, 'error');
    process.exit(1);
  }
}

async function runHealthChecks() {
  log('Running health checks...', 'info');
  
  const results = {};
  for (const service of Object.keys(CONFIG.services)) {
    results[service] = await checkHealth(service);
  }
  
  console.log('\nHealth Check Results:');
  for (const [service, isHealthy] of Object.entries(results)) {
    const status = isHealthy ? '✓' : '✗';
    const color = isHealthy ? chalk.green : chalk.red;
    console.log(`  ${color(`${status} ${service}`)}`);
  }
  
  const allHealthy = Object.values(results).every(Boolean);
  return allHealthy;
}

function printSuccessBanner() {
  console.log('\n' + chalk.green.bold('🎉 AI Code Review Platform is running!'));
  console.log(chalk.cyan('=' .repeat(50)));
  
  console.log('\n📱 Access URLs:');
  console.log(chalk.blue(`  Frontend:     http://localhost:${CONFIG.services.frontend.port}`));
  console.log(chalk.blue(`  Backend API:   http://localhost:${CONFIG.services.backend.port}`));
  console.log(chalk.blue(`  API Docs:      http://localhost:${CONFIG.services.backend.port}/docs`));
  console.log(chalk.blue(`  OpenAPI:       http://localhost:${CONFIG.services.backend.port}/openapi.json`));
  
  console.log('\n🔧 Development Tools:');
  console.log(chalk.blue(`  PostgreSQL:     localhost:${CONFIG.services.postgres.port}`));
  console.log(chalk.blue(`  Redis:          localhost:${CONFIG.services.redis.port}`));
  console.log(chalk.blue(`  Neo4j:          http://localhost:${CONFIG.services.neo4j.port}`));
  console.log(chalk.blue(`  Prometheus:     http://localhost:${CONFIG.services.prometheus.port}`));
  
  console.log('\n📝 Management Commands:');
  console.log(chalk.yellow(`  npm run down     - Stop all services`));
  console.log(chalk.yellow(`  npm run logs     - Show service logs`));
  console.log(chalk.yellow(`  npm run health   - Check service health`));
  console.log(chalk.yellow(`  npm run doctor   - System diagnostics`));
  
  console.log('\n' + chalk.cyan('=' .repeat(50)));
}

async function runDoctor() {
  console.log(chalk.blue.bold('🔍 System Diagnostics'));
  console.log(chalk.cyan('=' .repeat(30)));
  
  // Check Docker installation
  log('Checking Docker installation...', 'info');
  const dockerOk = await checkDockerInstallation();
  if (dockerOk) {
    log('Docker is properly installed', 'success');
  } else {
    log('Docker is not installed or not working', 'error');
    console.log(chalk.yellow('Install Docker from: https://docker.com/get-docker'));
  }
  
  // Check Docker Compose
  log('Checking Docker Compose...', 'info');
  try {
    await execCommand('docker', ['compose', 'version']);
    log('Docker Compose is working', 'success');
  } catch {
    log('Docker Compose is not working', 'error');
  }
  
  // Check Node.js version
  const nodeVersion = process.version;
  log(`Node.js version: ${nodeVersion}`, 'info');
  
  // Check available ports
  log('Checking port availability...', 'info');
  await checkPortConflicts();
  
  // Check docker-compose.yml exists
  const composeFile = join(process.cwd(), 'docker-compose.yml');
  if (existsSync(composeFile)) {
    log('docker-compose.yml found', 'success');
  } else {
    log('docker-compose.yml not found', 'error');
  }
  
  console.log(chalk.cyan('=' .repeat(30)));
}

async function clean() {
  const spinner = ora('Cleaning up Docker resources...').start();
  
  try {
    // Stop and remove containers
    await execCommand('docker', ['compose', 'down', '-v', '--remove-orphans']);
    
    // Remove unused images
    await execCommand('docker', ['image', 'prune', '-f']);
    
    // Remove unused volumes (with confirmation)
    const { confirmVolumeCleanup } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'confirmVolumeCleanup',
        message: 'Remove unused Docker volumes? This will delete data!',
        default: false,
      }
    ]);
    
    if (confirmVolumeCleanup) {
      await execCommand('docker', ['volume', 'prune', '-f']);
    }
    
    spinner.succeed('Cleanup completed');
  } catch (error) {
    spinner.fail('Cleanup failed');
    log(`Cleanup error: ${error.message}`, 'error');
  }
}

// Main command handlers
async function handleBuild() {
  console.log(chalk.blue.bold('🚀 Building and Starting AI Code Review Platform'));
  console.log(chalk.cyan('=' .repeat(50)));
  
  try {
    // Pre-flight checks
    log('Running pre-flight checks...', 'info');
    
    const dockerOk = await checkDockerInstallation();
    if (!dockerOk) {
      log('Docker is not properly installed', 'error');
      console.log(chalk.yellow('Please install Docker from: https://docker.com/get-docker'));
      process.exit(1);
    }
    
    const portsOk = await checkPortConflicts();
    if (!portsOk) {
      process.exit(1);
    }
    
    // Get profile from environment or prompt
    let profile = process.env.PROFILE || process.env.COMPOSE_PROFILES || 'dev';
    
    if (!process.env.PROFILE && !process.env.COMPOSE_PROFILES) {
      const { selectedProfile } = await inquirer.prompt([
        {
          type: 'list',
          name: 'selectedProfile',
          message: 'Select deployment profile:',
          choices: [
            { name: 'Development', value: 'dev' },
            { name: 'Production', value: 'prod' },
            { name: 'Staging', value: 'staging' },
          ],
          default: 'dev'
        }
      ]);
      profile = selectedProfile;
    }
    
    log(`Using profile: ${profile}`, 'info');
    
    // Build images
    const buildOk = await buildImages();
    if (!buildOk) {
      process.exit(1);
    }
    
    // Start services
    const startOk = await startServices(profile);
    if (!startOk) {
      process.exit(1);
    }
    
    // Wait for health checks
    log('Waiting for services to be healthy...', 'info');
    
    const coreServices = ['backend', 'frontend'];
    let allHealthy = true;
    
    for (const service of coreServices) {
      const healthy = await waitForHealth(service);
      if (!healthy) {
        allHealthy = false;
        break;
      }
    }
    
    if (allHealthy) {
      printSuccessBanner();
      log('Platform started successfully!', 'success');
    } else {
      log('Some services failed to start properly', 'error');
      console.log(chalk.yellow('\nTroubleshooting:'));
      console.log('1. Run "npm run logs" to see service logs');
      console.log('2. Run "npm run doctor" for system diagnostics');
      console.log('3. Check docker-compose.yml configuration');
      process.exit(1);
    }
    
  } catch (error) {
    log(`Unexpected error: ${error.message}`, 'error');
    process.exit(1);
  }
}

async function handleStart() {
  await startServices();
  await handleHealth();
}

async function handleDown() {
  await stopServices();
}

async function handleLogs() {
  await showLogs();
}

async function handleHealth() {
  const allHealthy = await runHealthChecks();
  process.exit(allHealthy ? 0 : 1);
}

async function handleDoctor() {
  await runDoctor();
}

async function handleClean() {
  await clean();
}

// CLI argument parsing
async function main() {
  const command = process.argv[2];
  
  switch (command) {
    case 'build':
      await handleBuild();
      break;
    case 'start':
      await handleStart();
      break;
    case 'down':
      await handleDown();
      break;
    case 'logs':
      await handleLogs();
      break;
    case 'health':
      await handleHealth();
      break;
    case 'doctor':
      await handleDoctor();
      break;
    case 'clean':
      await handleClean();
      break;
    default:
      console.log(chalk.blue.bold('AI Code Review Platform - One Command Runner'));
      console.log(chalk.cyan('=' .repeat(45)));
      console.log('\nUsage: npm run <command>');
      console.log('\nCommands:');
      console.log(chalk.green('  build    ') + 'Build images and start all services');
      console.log(chalk.green('  start    ') + 'Start services (if already built)');
      console.log(chalk.green('  down     ') + 'Stop and remove all services');
      console.log(chalk.green('  logs     ') + 'Show service logs');
      console.log(chalk.green('  health   ') + 'Check service health');
      console.log(chalk.green('  doctor   ') + 'Run system diagnostics');
      console.log(chalk.green('  clean    ') + 'Clean up Docker resources');
      console.log('\nExamples:');
      console.log(chalk.yellow('  npm run build     ') + '# Build and start everything');
      console.log(chalk.yellow('  npm run down       ') + '# Stop all services');
      console.log(chalk.yellow('  PROFILE=prod npm run build') + '# Build with production profile');
      break;
  }
}

// Handle process termination
process.on('SIGINT', () => {
  log('\nReceived interrupt signal. Exiting gracefully...', 'warning');
  process.exit(0);
});

process.on('SIGTERM', () => {
  log('\nReceived termination signal. Exiting gracefully...', 'warning');
  process.exit(0);
});

// Run main function
main().catch(error => {
  log(`Fatal error: ${error.message}`, 'error');
  process.exit(1);
});