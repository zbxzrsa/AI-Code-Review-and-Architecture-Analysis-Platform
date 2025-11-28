import React, { useState, useEffect } from 'react';
import { DocumentTextIcon, BookOpenIcon, CodeBracketIcon, CogIcon } from '@heroicons/react/24/outline';

interface VersionDocs {
  version: string;
  name: string;
  description: string;
  model: string;
  status: 'production' | 'experimental' | 'deprecated';
  metrics?: {
    avg_latency: string;
    accuracy: number;
    throughput: number;
    error_rate: number;
    cost_per_review: string;
  };
  features?: string[];
  limitations?: string[];
  api_info?: any;
}

interface DocsPageProps {
  onVersionChange?: (version: string) => void;
}

const DocsPage: React.FC<DocsPageProps> = ({ onVersionChange }) => {
  const [selectedVersion, setSelectedVersion] = useState('v1');
  const [versionDocs, setVersionDocs] = useState<Record<string, VersionDocs>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadVersionDocs = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Load documentation for all versions
        const versions = ['v1', 'v2', 'v3'];
        const docs: Record<string, VersionDocs> = {};
        
        for (const version of versions) {
          try {
            const response = await fetch(`/api/versions/config/${version}`);
            const config = await response.json();
            
            docs[version] = {
              version: config.version || version,
              name: config.name || `${version} Configuration`,
              description: config.description || `Configuration and documentation for ${version}`,
              model: config.model?.name || 'Unknown',
              status: config.status || 'unknown',
              metrics: config.metrics,
              features: config.features || [],
              limitations: config.limitations || [],
              api_info: config.api_info
            };
          } catch (err) {
            console.error(`Failed to load docs for ${version}:`, err);
            // Fallback documentation
            docs[version] = {
              version,
              name: `${version.toUpperCase()} Configuration`,
              description: `Configuration and documentation for ${version}`,
              model: 'Unknown',
              status: 'unknown' as any,
              metrics: {},
              features: [],
              limitations: ['Documentation unavailable']
            };
          }
        }
        
        setVersionDocs(docs);
      } catch (err) {
        setError('Failed to load version documentation');
        console.error('Documentation load error:', err);
      } finally {
        setLoading(false);
      }
    };

    loadVersionDocs();
  }, []);

  const handleVersionChange = (version: string) => {
    setSelectedVersion(version);
    if (onVersionChange) {
      onVersionChange(version);
    }
  };

  const getStatusBadge = (status: VersionDocs['status']) => {
    switch (status) {
      case 'production':
        return 'bg-green-100 text-green-800';
      case 'experimental':
        return 'bg-blue-100 text-blue-800';
      case 'deprecated':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: VersionDocs['status']) => {
    switch (status) {
      case 'production':
        return '✅';
      case 'experimental':
        return '🧪';
      case 'deprecated':
        return '⚠️';
      default:
        return '❓';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-900"></div>
          <p className="mt-4 text-gray-600">Loading documentation...</p>
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
              <BookOpenIcon className="h-6 w-6 mr-2" />
              <h1 className="text-2xl font-bold text-gray-900">Version Documentation</h1>
              <p className="ml-4 text-gray-500">Comprehensive documentation for all AI versions</p>
            </div>
            <select
              value={selectedVersion}
              onChange={(e) => handleVersionChange(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="v1">v1_stable</option>
              <option value="v2">v2_experimental</option>
              <option value="v3">v3_deprecated</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Version Overview */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <CogIcon className="h-6 w-6 mr-2" />
                Version Overview
              </h2>
              
              <div className="space-y-4">
                {Object.entries(versionDocs).map(([version, docs]) => (
                  <div key={version} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusBadge(docs.status)}`}>
                          {getStatusIcon(docs.status)}
                          <span className="ml-2">{docs.name}</span>
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">{version}</span>
                    </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium text-gray-900">Model</h4>
                        <p className="text-sm text-gray-600">{docs.model}</p>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-900">Status</h4>
                        <p className="text-sm text-gray-600 capitalize">{docs.status}</p>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-900">Description</h4>
                        <p className="text-sm text-gray-600">{docs.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h2>
              
              <div className="space-y-6">
                {versionDocs[selectedVersion]?.metrics && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-medium text-gray-900">Latency</h4>
                      <p className="text-2xl font-bold text-blue-600">
                        {versionDocs[selectedVersion].metrics.avg_latency}
                      </p>
                      <p className="text-sm text-gray-500">Average response time</p>
                    </div>
                    
                    <div className="space-y-2">
                      <h4 className="font-medium text-gray-900">Accuracy</h4>
                      <p className="text-2xl font-bold text-green-600">
                        {versionDocs[selectedVersion].metrics.accuracy}%
                      </p>
                      <p className="text-sm text-gray-500">Code analysis accuracy</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-medium text-gray-900">Throughput</h4>
                      <p className="text-2xl font-bold text-purple-600">
                        {versionDocs[selectedVersion].metrics.throughput}
                      </p>
                      <p className="text-sm text-gray-500">Reviews per minute</p>
                    </div>
                    
                    <div className="space-y-2">
                      <h4 className="font-medium text-gray-900">Error Rate</h4>
                      <p className="text-2xl font-bold text-red-600">
                        {versionDocs[selectedVersion].metrics.error_rate}%
                      </p>
                      <p className="text-sm text-gray-500">Analysis failure rate</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-900">Cost</h4>
                      <p className="text-2xl font-bold text-orange-600">
                        {versionDocs[selectedVersion].metrics.cost_per_review}
                      </p>
                      <p className="text-sm text-gray-500">Cost per review</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Features */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Features & Capabilities</h2>
              
              <div className="space-y-4">
                {versionDocs[selectedVersion]?.features && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Supported Features</h4>
                    <ul className="space-y-2">
                      {versionDocs[selectedVersion].features.map((feature, index) => (
                        <li key={index} className="flex items-start">
                          <CodeBracketIcon className="h-5 w-5 mr-2 text-green-500 mt-0.5" />
                          <span className="text-sm text-gray-700">{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {versionDocs[selectedVersion]?.limitations && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Limitations</h4>
                    <ul className="space-y-2">
                      {versionDocs[selectedVersion].limitations.map((limitation, index) => (
                        <li key={index} className="flex items-start">
                          <span className="text-red-500 mr-2">⚠️</span>
                          <span className="text-sm text-gray-700">{limitation}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* API Information */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">API Information</h2>
              
              <div className="space-y-4">
                {versionDocs[selectedVersion]?.api_info && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">API Configuration</h4>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <pre className="text-xs text-gray-700 overflow-x-auto">
                        {JSON.stringify(versionDocs[selectedVersion].api_info, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Usage Examples</h4>
                  <div className="space-y-3">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 mb-2">Basic Review Request</h5>
                      <pre className="text-xs text-gray-700">
{`curl -X POST http://localhost:8000/api/review \\
  -H "Content-Type: application/json" \\
  -d '{
    "code": "def example(): pass",
    "language": "python",
    "version": "${selectedVersion}",
    "focus_areas": ["security", "quality"]
  }'`}
                      </pre>
                    </div>
                    
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 mb-2">JavaScript Review</h5>
                      <pre className="text-xs text-gray-700">
{`curl -X POST http://localhost:8000/api/review \\
  -H "Content-Type: application/json" \\
  -d '{
    "code": "function example() { return true; }",
    "language": "javascript",
    "version": "${selectedVersion}",
    "focus_areas": ["security", "performance"]
  }'`}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocsPage;