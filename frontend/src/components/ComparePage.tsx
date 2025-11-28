import React, { useState, useEffect, useRef } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

interface VersionMetrics {
  version: string;
  name: string;
  avg_latency: number;
  accuracy: number;
  throughput: number;
  error_rate: number;
  cost_per_review: string;
}

interface ComparisonData {
  v1: VersionMetrics;
  v2: VersionMetrics;
  differences: {
    latency: number;
    accuracy: number;
    throughput: number;
    error_rate: number;
  };
  recommendation: string;
}

interface ComparePageProps {
  onVersionChange?: (version: string) => void;
}

const ComparePage: React.FC<ComparePageProps> = ({ onVersionChange }) => {
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedVersions, setSelectedVersions] = useState({ v1: 'v1', v2: 'v2' });

  useEffect(() => {
    const loadComparisonData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Load comparison from version router
        const response = await fetch('/api/versions/compare/v1/v2');
        const data = await response.json();

        setComparisonData(data);
      } catch (err) {
        setError('Failed to load comparison data');
        console.error('Comparison data load error:', err);
      } finally {
        setLoading(false);
      }
    };

    loadComparisonData();
  }, []);

  const handleVersionChange = (side: 'v1' | 'v2', version: string) => {
    setSelectedVersions(prev => ({ ...prev, [side]: version }));

    // Load new comparison data
    const loadNewComparison = async () => {
      try {
        const response = await fetch(
          `/api/versions/compare/${selectedVersions.v1}/${selectedVersions.v2}`
        );
        const data = await response.json();
        setComparisonData(data);
      } catch (err) {
        setError('Failed to load comparison data');
        console.error('Comparison data load error:', err);
      }
    };

    loadNewComparison();
  };

  const getImprovementColor = (value: number, inverse = false) => {
    if (inverse) {
      if (value > 0) return 'text-red-600';
      if (value < 0) return 'text-green-600';
      return 'text-gray-600';
    } else {
      if (value > 0) return 'text-green-600';
      if (value < 0) return 'text-red-600';
      return 'text-gray-600';
    }
  };

  const getImprovementIcon = (value: number) => {
    if (value > 0) return '↑';
    if (value < 0) return '↓';
    return '→';
  };

  const formatLatency = (latency: number) => {
    if (latency < 1) {
      return `${(latency * 1000).toFixed(0)}ms`;
    }
    return `${latency.toFixed(2)}s`;
  };

  const latencyData = comparisonData
    ? [
        {
          name: comparisonData.v1.name,
          latency: parseFloat(comparisonData.v1.avg_latency?.toString() || '0'),
          fill: '#3b82f6',
        },
        {
          name: comparisonData.v2.name,
          latency: parseFloat(comparisonData.v2.avg_latency?.toString() || '0'),
          fill: '#10b981',
        },
      ]
    : [];

  const accuracyData = comparisonData
    ? [
        {
          name: comparisonData.v1.name,
          accuracy: parseFloat((comparisonData.v1.accuracy * 100).toString() || '0'),
          fill: '#3b82f6',
        },
        {
          name: comparisonData.v2.name,
          accuracy: parseFloat((comparisonData.v2.accuracy * 100).toString() || '0'),
          fill: '#10b981',
        },
      ]
    : [];

  const throughputData = comparisonData
    ? [
        {
          name: comparisonData.v1.name,
          throughput: parseFloat(comparisonData.v1.throughput?.toString() || '0'),
          fill: '#3b82f6',
        },
        {
          name: comparisonData.v2.name,
          throughput: parseFloat(comparisonData.v2.throughput?.toString() || '0'),
          fill: '#10b981',
        },
      ]
    : [];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-900"></div>
          <p className="mt-4 text-gray-600">Loading comparison data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 text-xl mb-4">⚠️ Error</div>
          <p className="text-gray-600">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-semibold text-gray-900">Version Comparison</h1>
              <p className="ml-4 text-gray-500">Compare AI model performance across versions</p>
            </div>
            <button
              onClick={() => window.history.back()}
              className="text-gray-500 hover:text-gray-700"
            >
              <ArrowPathIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Version Selectors */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">Select Versions to Compare</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Version 1 (Baseline)
              </label>
              <select
                value={selectedVersions.v1}
                onChange={e => handleVersionChange('v1', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="v1">v1_stable (CodeBERT)</option>
                <option value="v2">v2_experimental (StarCoder)</option>
                <option value="v3">v3_deprecated (GPT-3.5)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Version 2 (Comparison)
              </label>
              <select
                value={selectedVersions.v2}
                onChange={e => handleVersionChange('v2', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="v1">v1_stable (CodeBERT)</option>
                <option value="v2">v2_experimental (StarCoder)</option>
                <option value="v3">v3_deprecated (GPT-3.5)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Comparison Results */}
      {comparisonData && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="text-2xl font-bold text-gray-900">
                {formatLatency(parseFloat(comparisonData.v2.avg_latency?.toString() || '0'))}
              </div>
              <div className="text-sm text-gray-500">v2 Latency</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="text-2xl font-bold text-gray-900">
                {formatLatency(parseFloat(comparisonData.v1.avg_latency?.toString() || '0'))}
              </div>
              <div className="text-sm text-gray-500">v1 Latency</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div
                className={`text-2xl font-bold ${getImprovementColor(comparisonData.differences.latency)}`}
              >
                {getImprovementIcon(comparisonData.differences.latency)}{' '}
                {Math.abs(comparisonData.differences.latency).toFixed(2)}s
              </div>
              <div className="text-sm text-gray-500">Latency Difference</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div
                className={`text-2xl font-bold ${getImprovementColor(comparisonData.differences.accuracy)}`}
              >
                {getImprovementIcon(comparisonData.differences.accuracy)}{' '}
                {(comparisonData.differences.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Accuracy Difference</div>
            </div>
          </div>

          {/* Recommendation */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-6 w-6 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-8 8 0 016 0zm-7 4a1 1 0 11-1 1H5a1 1 0 01-1-1V11a1 1 0 011-1 1h11a1 1 0 011 1v4a1 1 0 011-1-1h4a1 1 0 011-1-1v-4a1 1 0 011-1-1zM2 11a1 1 0 011-1h2a1 1 0 011-1v2a1 1 0 011-1h2a1 1 0 011-1v-2H3a1 1 0 011-1-1V9a1 1 0 011-1-1H2a1 1 0 011-1-1v-2z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-lg font-medium text-blue-900">Recommendation</h3>
                <p className="text-blue-700 mt-1">{comparisonData.recommendation}</p>
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Latency Chart */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Latency Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={value => formatLatency(value as number)} />
                  <Legend />
                  <Bar dataKey="latency" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Accuracy Chart */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Accuracy Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={value => `${value}%`} />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Throughput Chart */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Throughput Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={throughputData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="throughput" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Detailed Metrics Table */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 lg:col-span-2">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Metrics</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Metric
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {comparisonData.v1.name}
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {comparisonData.v2.name}
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Difference
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Average Latency
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatLatency(
                          parseFloat(comparisonData.v1.avg_latency?.toString() || '0')
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatLatency(
                          parseFloat(comparisonData.v2.avg_latency?.toString() || '0')
                        )}
                      </td>
                      <td
                        className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getImprovementColor(comparisonData.differences.latency)}`}
                      >
                        {getImprovementIcon(comparisonData.differences.latency)}{' '}
                        {formatLatency(Math.abs(comparisonData.differences.latency))}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Accuracy
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {((comparisonData.v1.accuracy || 0) * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {((comparisonData.v2.accuracy || 0) * 100).toFixed(1)}%
                      </td>
                      <td
                        className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getImprovementColor(comparisonData.differences.accuracy)}`}
                      >
                        {getImprovementIcon(comparisonData.differences.accuracy)}{' '}
                        {(comparisonData.differences.accuracy * 100).toFixed(1)}%
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Throughput
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {comparisonData.v1.throughput || 0} req/min
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {comparisonData.v2.throughput || 0} req/min
                      </td>
                      <td
                        className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getImprovementColor(comparisonData.differences.throughput)}`}
                      >
                        {getImprovementIcon(comparisonData.differences.throughput)}{' '}
                        {Math.abs(comparisonData.differences.throughput)} req/min
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Cost per Review
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {comparisonData.v1.cost_per_review}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {comparisonData.v2.cost_per_review}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">N/A</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ComparePage;
