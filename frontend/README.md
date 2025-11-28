# AI Code Review Platform - Frontend

A modern, responsive React dashboard for AI-powered code review and architecture analysis with version management capabilities.

## 🚀 Features

### Core Functionality

- **Version Management**: Switch between v1_stable, v2_experimental, and v3_deprecated AI models
- **Code Analysis**: Upload and analyze code with Monaco Editor
- **Review Results**: Comprehensive issue tracking with severity-based visualization
- **Version Comparison**: Side-by-side performance metrics with D3.js charts
- **Documentation**: Auto-generated version-specific documentation
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Dark Mode**: Toggle between light and dark themes
- **Offline Support**: LocalStorage caching for improved performance

## 🛠 Technology Stack

- **React 18** with TypeScript for type safety
- **Tailwind CSS** for utility-first styling
- **Monaco Editor** for professional code editing
- **Recharts** for interactive data visualization
- **React Router** for client-side navigation
- **Heroicons** for consistent iconography

## 📁 Component Architecture

```
src/
├── components/
│   ├── VersionSelector.tsx          # Version dropdown with status indicators
│   ├── MonacoEditor.tsx           # Code editor with syntax highlighting
│   ├── ReviewResults.tsx         # Issue display with severity colors
│   ├── ComparePage.tsx           # Version comparison with charts
│   ├── HomePage.tsx              # Main dashboard with code upload
│   └── DocsPage.tsx              # Version documentation
│   └── App.tsx                  # Main app with routing
├── hooks/                     # Custom React hooks
├── types/                     # TypeScript type definitions
└── utils/                     # Utility functions
└── styles/                    # Global styles and Tailwind config
```

## 🎨 Key Components

### VersionSelector

- Dropdown to select AI version (v1/v2/v3)
- Visual status indicators (production/experimental/deprecated)
- Real-time metrics display (latency, accuracy, throughput)
- Smooth transitions and hover states

### MonacoEditor

- Full-featured code editor with syntax highlighting
- Support for 40+ programming languages
- Keyboard shortcuts (Ctrl+S for format, Ctrl+F for find)
- Theme-aware (light/dark mode support)
- Loading states and error handling

### ReviewResults

- Severity-based issue categorization and coloring
- Grouped issues by type (security, quality, performance, style)
- Expandable code snippets with syntax highlighting
- Confidence scores and processing metrics
- Caching indicators and performance stats

### ComparePage

- Side-by-side version comparison
- Interactive D3.js charts (bar charts for metrics)
- Performance difference visualization
- Detailed metrics table
- Recommendation engine based on comparison results

### HomePage

- Drag-and-drop file upload
- Monaco editor integration
- Focus area selection (security, quality, performance)
- Sample code library for quick testing
- Real-time analysis progress indicators
- Version-aware analysis results

## 🎨 Design System

### Responsive Design

- **Mobile-first approach** with breakpoints:
  - Small: < 640px
  - Medium: 640px - 1024px
  - Large: 1024px - 1280px
  - XL: 1280px+
- **Flexible grid layouts** that adapt to screen size

### Accessibility (WCAG AA)

- Semantic HTML5 structure
- ARIA labels and roles
- Keyboard navigation support
- High contrast ratios
- Focus indicators and skip links
- Screen reader compatibility

### Dark Mode

- System-wide theme toggle with localStorage persistence
- Carefully crafted color palette for both modes
- Smooth transitions between themes
- Component-level theme awareness

## 🚀 Performance Optimizations

### Code Splitting

- Lazy loading with React.lazy()
- Route-based code splitting
- Dynamic imports for reduced bundle size
- Preloading strategies for critical components

### Caching Strategy

- LocalStorage for API responses
- Component-level memoization with React.memo
- Service worker for background caching
- Offline-first approach with fallbacks

### Bundle Optimization

- Tree shaking for unused code elimination
- Minification in production builds
- Asset optimization and compression
- CDN delivery for static assets

## 🔧 Development Workflow

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview build
npm run preview
```

### Environment Variables

```bash
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Feature Flags
VITE_ENABLE_DARK_MODE=true
VITE_ENABLE_ANALYTICS=true
```

## 📱 API Integration

### Backend Endpoints

- `/api/versions/status` - Version status and metrics
- `/api/versions/compare/v1/v2` - Version comparison
- `/api/versions/config/{version}` - Version-specific configuration
- `/api/review` - Code analysis endpoint
- `/api/health` - Health check endpoint

### Request/Response Format

```typescript
interface ReviewRequest {
  code: string;
  language: string;
  version?: string;
  focus_areas?: string[];
}

interface ReviewResponse {
  issues: Issue[];
  score: number;
  processing_time: number;
  version: string;
  model_used: string;
  cached: boolean;
}
```

## 🎯 State Management

### Local Storage Strategy

- Version preference persistence
- Theme preference with system detection
- Analysis result caching
- User session data
- Error handling and retry logic

### Error Boundaries

- Graceful degradation when services are unavailable
- User-friendly error messages
- Automatic retry with exponential backoff
- Fallback to cached results when possible

## 🔐 Testing Strategy

### Component Testing

- Jest and React Testing Library for unit tests
- Component storybooks with Storybook
- Mock service workers for API testing
- Visual regression testing with Percy

### Integration Testing

- End-to-end test coverage with Playwright
- API integration testing with MSW
- Performance testing with Lighthouse CI
- Accessibility testing with axe-core

## 📚 Monitoring

### Performance Metrics

- Core Web Vitals tracking
- Bundle analysis and optimization
- Error rate monitoring
- User interaction analytics
- Performance budget enforcement

### Error Tracking

- Sentry integration for error reporting
- Custom error boundaries with detailed context
- Performance impact assessment
- User feedback collection

## 🚀 Deployment

### Build Process

- Optimized production builds with Vite
- Static asset optimization and compression
- Environment-specific configuration
- Docker containerization support
- CI/CD pipeline integration

### Production Considerations

- CDN configuration for static assets
- API rate limiting and caching
- Security headers and CSP configuration
- Health checks and monitoring setup
- Scalability and load balancing

---

## 🎯 Getting Started

### Prerequisites

- Node.js 18+ and npm 8+
- Modern web browser with ES6+ support
- Git for version control

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-code-review-platform

# Install dependencies
npm install

# Start development
npm run dev

# Open in browser
# Navigate to http://localhost:3000
```

### Configuration

1. Copy `.env.example` to `.env`
2. Configure API endpoints
3. Set up feature flags as needed
4. Run `npm run dev` to start

---

## 📚 Contributing

### Development Guidelines

1. Follow the established code patterns and conventions
2. Write comprehensive tests for new features
3. Update documentation for API changes
4. Ensure accessibility compliance
5. Performance test all changes

### Code Style

- Use TypeScript for all new code
- Follow React hooks best practices
- Implement proper error handling
- Write semantic, accessible HTML
- Use Tailwind classes for styling

### Pull Request Process

1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit pull request with detailed description
5. Ensure CI/CD passes

---

## 📄 Architecture Decisions

### Component Architecture

- **Container Components**: Self-contained, reusable components
- **Custom Hooks**: Shared logic extracted into custom hooks
- **Utility Functions**: Pure functions for common operations
- **Type Safety**: Comprehensive TypeScript coverage

### State Management

- **Local State**: useState for component-level state
- **Server State**: React Query for server state
- **Caching**: LocalStorage for offline support

### Routing Strategy

- **Client-Side**: React Router for SPA navigation
- **Lazy Loading**: Code splitting at route level
- **Error Boundaries**: 404 handling with Navigate component

### Performance Patterns

- **Memoization**: React.memo for expensive components
- **Virtualization**: React.memo for heavy computations
- **Debouncing**: User input debouncing for search/filter operations

---

This frontend provides a modern, accessible, and performant user interface for the AI Code Review Platform, with comprehensive version management, real-time analysis, and responsive design capabilities.
