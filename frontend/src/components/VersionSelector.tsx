import React, { useState, useEffect } from 'react';
import { ChevronDownIcon } from '@heroicons/react/24/outline';

interface Version {
  id: string;
  name: string;
  description: string;
  status: 'production' | 'experimental' | 'deprecated';
  model: string;
  metrics?: {
    avg_latency: string;
    accuracy: number;
    throughput: number;
    error_rate: number;
  };
}

interface VersionSelectorProps {
  selectedVersion: string;
  onVersionChange: (version: string) => void;
  disabled?: boolean;
}

const VersionSelector: React.FC<VersionSelectorProps> = ({
  selectedVersion,
  onVersionChange,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [versions, setVersions] = useState<Version[]>([]);

  useEffect(() => {
    // Load version configurations
    const loadVersions = async () => {
      try {
        const response = await fetch('/api/versions/status');
        const data = await response.json();

        const versionData: Version[] = [
          {
            id: 'v1',
            name: 'v1_stable',
            description: 'Production-ready CodeBERT with proven reliability',
            status: 'production',
            model: 'microsoft/codebert-base',
            metrics: data.versions?.v1?.config?.metrics,
          },
          {
            id: 'v2',
            name: 'v2_experimental',
            description: 'Experimental StarCoder with advanced features',
            status: 'experimental',
            model: 'bigcode/starcoder',
            metrics: data.versions?.v2?.config?.metrics,
          },
          {
            id: 'v3',
            name: 'v3_deprecated',
            description: 'Deprecated GPT-3.5 (high cost, slow)',
            status: 'deprecated',
            model: 'gpt-3.5-turbo',
            metrics: data.versions?.v3?.config?.metrics,
          },
        ];

        setVersions(versionData);
      } catch (error) {
        console.error('Failed to load versions:', error);
        // Fallback versions
        setVersions([
          {
            id: 'v1',
            name: 'v1_stable',
            description: 'Production-ready CodeBERT',
            status: 'production',
            model: 'microsoft/codebert-base',
          },
          {
            id: 'v2',
            name: 'v2_experimental',
            description: 'Experimental StarCoder',
            status: 'experimental',
            model: 'bigcode/starcoder',
          },
          {
            id: 'v3',
            name: 'v3_deprecated',
            description: 'Deprecated GPT-3.5',
            status: 'deprecated',
            model: 'gpt-3.5-turbo',
          },
        ]);
      }
    };

    loadVersions();
  }, []);

  const getStatusColor = (status: Version['status']) => {
    switch (status) {
      case 'production':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'experimental':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'deprecated':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status: Version['status']) => {
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

  const selectedVersionData = versions.find(v => v.id === selectedVersion);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className={`
          flex items-center justify-between w-full px-4 py-2 text-sm font-medium
          bg-white border border-gray-300 rounded-lg shadow-sm
          hover:bg-gray-50 focus:outline-none focus:ring-2 
          focus:ring-blue-500 focus:border-blue-500
          transition-colors duration-200
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="text-lg">
              {selectedVersionData ? getStatusIcon(selectedVersionData.status) : '❓'}
            </span>
            <span className="font-medium">
              {selectedVersionData ? selectedVersionData.name : 'Select Version'}
            </span>
          </div>
          {selectedVersionData?.metrics && (
            <div className="flex items-center space-x-4 text-xs text-gray-500">
              <span>⚡ {selectedVersionData.metrics.avg_latency}</span>
              <span>🎯 {selectedVersionData.metrics.accuracy}%</span>
            </div>
          )}
        </div>
        <ChevronDownIcon
          className={`h-5 w-5 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <div className="absolute z-10 mt-1 w-full bg-white border border-gray-300 rounded-lg shadow-lg">
          <div className="py-1">
            {versions.map(version => (
              <button
                key={version.id}
                onClick={() => {
                  onVersionChange(version.id);
                  setIsOpen(false);
                }}
                className={`
                  w-full px-4 py-3 text-left hover:bg-gray-50
                  border-b border-gray-100 last:border-b-0
                  transition-colors duration-150
                  ${selectedVersion === version.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'}
                `}
                role="option"
                aria-selected={selectedVersion === version.id}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-lg">{getStatusIcon(version.status)}</span>
                      <span className="font-semibold">{version.name}</span>
                      <span
                        className={`
                        px-2 py-1 text-xs font-medium rounded-full
                        ${getStatusColor(version.status)}
                      `}
                      >
                        {version.status}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{version.description}</p>
                    <div className="text-xs text-gray-500 space-x-1">
                      <span>Model: {version.model}</span>
                      {version.metrics && (
                        <>
                          <span>•</span>
                          <span>Latency: {version.metrics.avg_latency}</span>
                          <span>•</span>
                          <span>Accuracy: {version.metrics.accuracy}%</span>
                        </>
                      )}
                    </div>
                  </div>
                  {selectedVersion === version.id && (
                    <div className="ml-2">
                      <svg
                        className="h-5 w-5 text-blue-600"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 1.414l8-8a1 1 0 011.414-1.414z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VersionSelector;
