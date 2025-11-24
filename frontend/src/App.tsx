import React, { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';

// Context providers
import { AuthProvider } from './contexts/AuthContext';
import { FeedbackProvider } from './components/feedback/FeedbackProvider';
import { SimpleThemeProvider } from './app/providers/SimpleThemeProvider';

// Components
import AppShell from './components/layout/AppShell';
import PrivateRoute from './components/PrivateRoute';
import OnboardingManager from './components/ui/OnboardingManager';

// Utils and routes
import initKeyboardNavigation from './utils/keyboardNavigation';
import { appRoutes } from './routes/appRoutes';

// Styles
import './styles/globalStyles.css';
import './styles/a11y.css';
import './styles/responsive-touch.css';

const AppContent = () => {
    useEffect(() => {
        initKeyboardNavigation();
    }, []);

    return (
        <Routes>
            {appRoutes.map((route) => {
                const Component = route.component;
                const element = <Component />;
                const protectedElement =
                    route.requiresAuth === false ? element : <PrivateRoute>{element}</PrivateRoute>;

                return <Route key={route.path} path={route.path} element={protectedElement} />;
            })}
        </Routes>
    );
};

const App = () => {
    return (
        <AuthProvider>
            <SimpleThemeProvider>
                <FeedbackProvider>
                    <AppShell>
                        <AppContent />
                        <OnboardingManager />
                    </AppShell>
                </FeedbackProvider>
            </SimpleThemeProvider>
        </AuthProvider>
    );
};

export default App;
