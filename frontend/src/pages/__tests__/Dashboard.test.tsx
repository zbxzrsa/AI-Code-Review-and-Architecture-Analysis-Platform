/**
 * Dashboard component tests (English only)
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Dashboard from '../Dashboard';

describe('Dashboard component', () => {
  test('renders the page title', () => {
    render(<Dashboard />);
    expect(screen.getByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
  });

  test('renders the statistic cards', () => {
    render(<Dashboard />);
    expect(screen.getByText('Analyzed Projects')).toBeInTheDocument();
    expect(screen.getByText('Issues Found')).toBeInTheDocument();
    expect(screen.getByText('Security Vulnerabilities')).toBeInTheDocument();
  });

  test('renders the expected metric values', () => {
    render(<Dashboard />);
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  test('renders the icons for each card', () => {
    const { container } = render(<Dashboard />);
    expect(container.querySelector('.anticon-code')).toBeInTheDocument();
    expect(container.querySelector('.anticon-bug')).toBeInTheDocument();
    expect(container.querySelector('.anticon-safety')).toBeInTheDocument();
  });

  test('applies the expected grid layout', () => {
    const { container } = render(<Dashboard />);
    const row = container.querySelector('.ant-row');
    expect(row).toBeInTheDocument();
    expect(row).toHaveAttribute('style', expect.stringContaining('margin-left: -8px'));

    const cols = container.querySelectorAll('.ant-col');
    expect(cols).toHaveLength(3);
    cols.forEach((col) => {
      expect(col).toHaveClass('ant-col-8');
    });
  });

  test('renders statistic components inside each card', () => {
    const { container } = render(<Dashboard />);
    const cards = container.querySelectorAll('.ant-card');
    expect(cards).toHaveLength(3);
    cards.forEach((card) => {
      expect(card.querySelector('.ant-statistic')).toBeInTheDocument();
    });
  });

  test('displays icons inside statistic prefixes', () => {
    const { container } = render(<Dashboard />);
    const statistics = container.querySelectorAll('.ant-statistic');
    expect(statistics).toHaveLength(3);

    statistics.forEach((statistic) => {
      const prefix = statistic.querySelector('.ant-statistic-content-prefix');
      expect(prefix).toBeInTheDocument();
      expect(prefix?.querySelector('.anticon')).toBeInTheDocument();
    });
  });

  test('applies responsive gutter styles', () => {
    const { container } = render(<Dashboard />);
    const row = container.querySelector('.ant-row');
    expect(row).toHaveStyle('margin-left: -8px; margin-right: -8px;');

    const cols = container.querySelectorAll('.ant-col');
    cols.forEach((col) => {
      expect(col).toHaveStyle('padding-left: 8px; padding-right: 8px;');
    });
  });
});