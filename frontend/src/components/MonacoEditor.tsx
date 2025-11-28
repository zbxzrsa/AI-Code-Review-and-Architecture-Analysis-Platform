import React, { useRef, useEffect, useState } from 'react';
import * as monaco from 'monaco-editor';

interface MonacoEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  height?: string;
  theme?: 'light' | 'dark';
  readOnly?: boolean;
  placeholder?: string;
  onMount?: (editor: monaco.editor.IStandaloneCodeEditor) => void;
}

const MonacoEditor: React.FC<MonacoEditorProps> = ({
  value,
  onChange,
  language = 'python',
  height = '400px',
  theme = 'light',
  readOnly = false,
  placeholder = '',
  onMount,
}) => {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isEditorReady, setIsEditorReady] = useState(false);

  useEffect(() => {
    // Initialize Monaco Editor
    if (containerRef.current && !editorRef.current) {
      // Configure Monaco
      monaco.editor.defineTheme('custom-dark', {
        base: 'vs-dark',
        inherit: true,
        rules: [
          { background: '1e1e1e' },
          { foreground: 'd4d4d4' },
          { 'editor.lineHighlightBackground': '#2d2d30' },
          { 'editorLineNumber.foreground': '#858585' },
          { 'editor.selectionBackground': '#264f78' },
          { 'editor.inactiveSelectionBackground': '#3a3d41' },
          { 'editor.wordHighlightBackground': '#264f78' },
          { 'editor.wordHighlightStrongBackground': '#1f2937' },
          { 'editorCursor.foreground': '#aeafaf' },
          { 'editorWhitespace.foreground': '#404040' },
          { 'editorIndentGuide.background': '#404040' },
          { 'editorIndentGuide.activeBackground': '#707070' },
          { 'editorIndentGuide.activeForeground': '#c8c8c8' },
        ],
      });

      // Create editor
      const editor = monaco.editor.create(containerRef.current, {
        value: value || '',
        language: language,
        theme: theme === 'dark' ? 'custom-dark' : 'vs',
        automaticLayout: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        fontSize: 14,
        fontFamily: 'Monaco, Consolas, "Courier New", monospace',
        lineNumbers: 'on',
        roundedSelection: false,
        cursorStyle: 'line',
        wordWrap: 'on',
        bracketPairColorization: { enabled: true },
        suggest: {
          showKeywords: true,
          showSnippets: true,
        },
        quickSuggestions: {
          other: true,
          comments: true,
          strings: true,
        },
        readOnly: readOnly,
        placeholder: placeholder,
      });

      editorRef.current = editor;

      // Add change listener
      const disposable = editor.onDidChangeModelContent(() => {
        const newValue = editor.getValue();
        onChange(newValue);
      });

      // Add keyboard shortcuts
      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
        editor.getAction('editor.action.formatDocument')?.run();
      });

      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyF, () => {
        editor.getAction('editor.action.startFindReplaceAction')?.run();
      });

      setIsEditorReady(true);

      // Call onMount callback
      if (onMount) {
        onMount(editor);
      }

      // Cleanup
      return () => {
        disposable.dispose();
      };
    }
  }, [language, theme, readOnly, placeholder]);

  // Update value when prop changes
  useEffect(() => {
    if (editorRef.current && value !== editorRef.current.getValue()) {
      editorRef.current.setValue(value);
    }
  }, [value]);

  return (
    <div className="border border-gray-300 rounded-lg overflow-hidden">
      {/* Loading state */}
      {!isEditorReady && (
        <div className="flex items-center justify-center h-96 bg-gray-50">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            <p className="mt-2 text-sm text-gray-600">Loading editor...</p>
          </div>
        </div>
      )}

      {/* Editor container */}
      <div ref={containerRef} style={{ height }} className="monaco-editor-container" />

      {/* Editor toolbar */}
      {isEditorReady && (
        <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200">
          <div className="flex items-center space-x-2 text-xs text-gray-500">
            <span>Language: {language}</span>
            <span>•</span>
            <span>Ctrl+S: Format</span>
            <span>•</span>
            <span>Ctrl+F: Find & Replace</span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => {
                if (editorRef.current) {
                  const currentTheme = editorRef.current.getOption(
                    monaco.editor.EditorOption.theme
                  );
                  const newTheme = currentTheme === 'vs' ? 'custom-dark' : 'vs';
                  monaco.editor.setTheme(newTheme);
                }
              }}
              className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      )}

      <style jsx>{`
        .monaco-editor-container {
          border-radius: 0;
        }

        .monaco-editor-container .monaco-editor {
          border-radius: 0;
        }

        .monaco-editor-container .margin {
          margin: 0;
        }

        .monaco-editor-container .monaco-editor .inputarea {
          margin: 0;
        }

        .monaco-editor-container .monaco-editor .overflow-guard {
          border-radius: 0;
        }
      `}</style>
    </div>
  );
};

export default MonacoEditor;
