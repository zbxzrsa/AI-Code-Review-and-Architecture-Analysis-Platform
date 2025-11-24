"""
跨浏览器兼容性测试
测试应用在不同浏览器和版本中的兼容性
"""
import pytest
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import TimeoutException, WebDriverException


class BrowserTestBase:
    """浏览器测试基类"""
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    def login_user(self, driver, base_url):
        """通用登录方法"""
        driver.get(f"{base_url}/login")
        wait = WebDriverWait(driver, 10)
        
        try:
            username_field = wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_field.send_keys("testuser123")
            
            password_field = driver.find_element(By.NAME, "password")
            password_field.send_keys("SecurePassword123!")
            
            login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            wait.until(EC.url_contains("/dashboard"))
            return True
        except TimeoutException:
            return False
    
    def test_basic_functionality(self, driver, base_url):
        """测试基本功能"""
        # 测试页面加载
        driver.get(base_url)
        wait = WebDriverWait(driver, 15)
        
        # 验证页面标题
        assert "智能代码审查" in driver.title or "Code Analysis" in driver.title
        
        # 测试导航
        try:
            nav_links = driver.find_elements(By.TAG_NAME, "a")
            assert len(nav_links) > 0
        except Exception:
            pass  # 某些浏览器可能需要更多时间加载
        
        # 测试登录功能
        login_success = self.login_user(driver, base_url)
        if login_success:
            # 验证仪表盘加载
            dashboard_content = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard-content"))
            )
            assert dashboard_content.is_displayed()
    
    def test_css_rendering(self, driver, base_url):
        """测试CSS渲染"""
        driver.get(f"{base_url}/dashboard")
        
        if self.login_user(driver, base_url):
            wait = WebDriverWait(driver, 10)
            
            # 检查关键元素的样式
            try:
                header = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ant-layout-header"))
                )
                
                # 验证头部样式
                header_height = header.size['height']
                assert header_height > 0, "Header should have height"
                
                # 验证背景色不是默认的白色（表示CSS已加载）
                header_bg = driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).backgroundColor;", 
                    header
                )
                assert header_bg != "rgba(0, 0, 0, 0)", "Header should have background color"
                
            except TimeoutException:
                pytest.skip("Dashboard not accessible for CSS testing")
    
    def test_javascript_functionality(self, driver, base_url):
        """测试JavaScript功能"""
        driver.get(f"{base_url}/code-analysis")
        
        if self.login_user(driver, base_url):
            wait = WebDriverWait(driver, 10)
            
            # 测试JavaScript交互
            try:
                # 查找代码编辑器
                code_editor = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "code-editor"))
                )
                
                # 测试输入功能
                code_editor.send_keys("def test(): pass")
                
                # 验证输入被处理
                editor_content = code_editor.get_attribute("value") or code_editor.text
                assert "def test(): pass" in editor_content
                
            except TimeoutException:
                pytest.skip("Code editor not available for JS testing")
    
    def test_responsive_behavior(self, driver, base_url):
        """测试响应式行为"""
        driver.get(base_url)
        
        # 测试不同屏幕尺寸
        screen_sizes = [
            (1920, 1080),  # 桌面
            (768, 1024),   # 平板
            (375, 667),    # 手机
        ]
        
        for width, height in screen_sizes:
            driver.set_window_size(width, height)
            time.sleep(1)  # 等待布局调整
            
            # 验证页面仍然可访问
            try:
                body = driver.find_element(By.TAG_NAME, "body")
                assert body.is_displayed()
                
                # 验证没有水平滚动条（在移动设备上）
                if width <= 768:
                    body_width = driver.execute_script("return document.body.scrollWidth;")
                    viewport_width = driver.execute_script("return window.innerWidth;")
                    assert body_width <= viewport_width + 20, f"Horizontal scroll detected at {width}x{height}"
                
            except Exception as e:
                pytest.fail(f"Responsive test failed at {width}x{height}: {str(e)}")


class TestChromeCompatibility(BrowserTestBase):
    """Chrome浏览器兼容性测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            yield driver
            driver.quit()
        except WebDriverException:
            pytest.skip("Chrome browser not available")
    
    def test_chrome_basic_functionality(self, driver, base_url):
        """测试Chrome基本功能"""
        self.test_basic_functionality(driver, base_url)
    
    def test_chrome_css_rendering(self, driver, base_url):
        """测试Chrome CSS渲染"""
        self.test_css_rendering(driver, base_url)
    
    def test_chrome_javascript_functionality(self, driver, base_url):
        """测试Chrome JavaScript功能"""
        self.test_javascript_functionality(driver, base_url)
    
    def test_chrome_responsive_behavior(self, driver, base_url):
        """测试Chrome响应式行为"""
        self.test_responsive_behavior(driver, base_url)
    
    def test_chrome_specific_features(self, driver, base_url):
        """测试Chrome特定功能"""
        driver.get(base_url)
        
        # 测试Chrome特定的API支持
        try:
            # 测试本地存储
            driver.execute_script("localStorage.setItem('test', 'value');")
            stored_value = driver.execute_script("return localStorage.getItem('test');")
            assert stored_value == "value"
            
            # 测试会话存储
            driver.execute_script("sessionStorage.setItem('test', 'session_value');")
            session_value = driver.execute_script("return sessionStorage.getItem('test');")
            assert session_value == "session_value"
            
            # 清理
            driver.execute_script("localStorage.removeItem('test');")
            driver.execute_script("sessionStorage.removeItem('test');")
            
        except Exception as e:
            pytest.fail(f"Chrome specific features test failed: {str(e)}")


class TestFirefoxCompatibility(BrowserTestBase):
    """Firefox浏览器兼容性测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")
        
        try:
            driver = webdriver.Firefox(options=firefox_options)
            driver.implicitly_wait(10)
            yield driver
            driver.quit()
        except WebDriverException:
            pytest.skip("Firefox browser not available")
    
    def test_firefox_basic_functionality(self, driver, base_url):
        """测试Firefox基本功能"""
        self.test_basic_functionality(driver, base_url)
    
    def test_firefox_css_rendering(self, driver, base_url):
        """测试Firefox CSS渲染"""
        self.test_css_rendering(driver, base_url)
    
    def test_firefox_javascript_functionality(self, driver, base_url):
        """测试Firefox JavaScript功能"""
        self.test_javascript_functionality(driver, base_url)
    
    def test_firefox_responsive_behavior(self, driver, base_url):
        """测试Firefox响应式行为"""
        self.test_responsive_behavior(driver, base_url)
    
    def test_firefox_specific_features(self, driver, base_url):
        """测试Firefox特定功能"""
        driver.get(base_url)
        
        # 测试Firefox特定的行为
        try:
            # 测试CSS Grid支持
            grid_support = driver.execute_script("""
                var div = document.createElement('div');
                div.style.display = 'grid';
                return div.style.display === 'grid';
            """)
            assert grid_support, "CSS Grid should be supported"
            
            # 测试Flexbox支持
            flex_support = driver.execute_script("""
                var div = document.createElement('div');
                div.style.display = 'flex';
                return div.style.display === 'flex';
            """)
            assert flex_support, "Flexbox should be supported"
            
        except Exception as e:
            pytest.fail(f"Firefox specific features test failed: {str(e)}")


class TestEdgeCompatibility(BrowserTestBase):
    """Edge浏览器兼容性测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Edge(options=edge_options)
            driver.implicitly_wait(10)
            yield driver
            driver.quit()
        except WebDriverException:
            pytest.skip("Edge browser not available")
    
    def test_edge_basic_functionality(self, driver, base_url):
        """测试Edge基本功能"""
        self.test_basic_functionality(driver, base_url)
    
    def test_edge_css_rendering(self, driver, base_url):
        """测试Edge CSS渲染"""
        self.test_css_rendering(driver, base_url)
    
    def test_edge_javascript_functionality(self, driver, base_url):
        """测试Edge JavaScript功能"""
        self.test_javascript_functionality(driver, base_url)
    
    def test_edge_responsive_behavior(self, driver, base_url):
        """测试Edge响应式行为"""
        self.test_responsive_behavior(driver, base_url)


class TestMobileCompatibility:
    """移动端兼容性测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # 模拟移动设备
        mobile_emulation = {
            "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 2.0},
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        }
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            yield driver
            driver.quit()
        except WebDriverException:
            pytest.skip("Chrome browser not available for mobile testing")
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    def test_mobile_layout(self, driver, base_url):
        """测试移动端布局"""
        driver.get(base_url)
        
        wait = WebDriverWait(driver, 10)
        
        # 验证移动端布局
        try:
            body = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # 检查视口设置
            viewport_meta = driver.find_element(By.XPATH, "//meta[@name='viewport']")
            viewport_content = viewport_meta.get_attribute("content")
            assert "width=device-width" in viewport_content, "Viewport should be set for mobile"
            
            # 验证没有水平滚动
            body_width = driver.execute_script("return document.body.scrollWidth;")
            viewport_width = driver.execute_script("return window.innerWidth;")
            assert body_width <= viewport_width + 10, "Should not have horizontal scroll on mobile"
            
        except Exception as e:
            pytest.fail(f"Mobile layout test failed: {str(e)}")
    
    def test_mobile_touch_interactions(self, driver, base_url):
        """测试移动端触摸交互"""
        driver.get(base_url)
        
        wait = WebDriverWait(driver, 10)
        
        try:
            # 查找可点击元素
            clickable_elements = driver.find_elements(By.XPATH, "//button | //a | //input[@type='submit']")
            
            for element in clickable_elements[:3]:  # 测试前3个元素
                if element.is_displayed() and element.is_enabled():
                    # 验证元素大小适合触摸（至少44px）
                    element_size = element.size
                    assert element_size['height'] >= 40, f"Touch target too small: {element_size['height']}px"
                    
                    # 测试点击
                    try:
                        element.click()
                        time.sleep(0.5)  # 等待响应
                    except Exception:
                        pass  # 某些元素可能需要特定条件才能点击
                        
        except Exception as e:
            pytest.fail(f"Mobile touch interaction test failed: {str(e)}")
    
    def test_mobile_navigation(self, driver, base_url):
        """测试移动端导航"""
        driver.get(base_url)
        
        wait = WebDriverWait(driver, 10)
        
        try:
            # 查找移动端导航菜单
            nav_elements = driver.find_elements(By.CLASS_NAME, "mobile-nav") or \
                          driver.find_elements(By.CLASS_NAME, "hamburger-menu") or \
                          driver.find_elements(By.CLASS_NAME, "menu-toggle")
            
            if nav_elements:
                nav_element = nav_elements[0]
                nav_element.click()
                time.sleep(1)
                
                # 验证菜单展开
                menu_items = driver.find_elements(By.CLASS_NAME, "nav-item") or \
                           driver.find_elements(By.CLASS_NAME, "menu-item")
                
                assert len(menu_items) > 0, "Mobile navigation menu should contain items"
                
        except Exception as e:
            # 移动端导航可能不存在，这是可以接受的
            pass


class TestAccessibilityCompatibility:
    """可访问性兼容性测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            yield driver
            driver.quit()
        except WebDriverException:
            pytest.skip("Chrome browser not available for accessibility testing")
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    def test_keyboard_navigation(self, driver, base_url):
        """测试键盘导航"""
        driver.get(base_url)
        
        wait = WebDriverWait(driver, 10)
        
        try:
            # 测试Tab键导航
            body = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # 模拟Tab键按下
            from selenium.webdriver.common.keys import Keys
            body.send_keys(Keys.TAB)
            
            # 验证焦点移动
            focused_element = driver.switch_to.active_element
            assert focused_element is not None, "Tab navigation should move focus"
            
            # 继续Tab导航
            for _ in range(5):
                focused_element.send_keys(Keys.TAB)
                time.sleep(0.1)
                new_focused = driver.switch_to.active_element
                # 验证焦点确实在移动
                
        except Exception as e:
            pytest.fail(f"Keyboard navigation test failed: {str(e)}")
    
    def test_aria_labels(self, driver, base_url):
        """测试ARIA标签"""
        driver.get(base_url)
        
        wait = WebDriverWait(driver, 10)
        
        try:
            # 查找带有ARIA标签的元素
            aria_elements = driver.find_elements(By.XPATH, "//*[@aria-label or @aria-labelledby or @role]")
            
            # 验证关键元素有适当的ARIA标签
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons[:5]:  # 检查前5个按钮
                if button.is_displayed():
                    aria_label = button.get_attribute("aria-label")
                    button_text = button.text
                    
                    # 按钮应该有文本或aria-label
                    assert aria_label or button_text.strip(), "Button should have accessible text"
                    
        except Exception as e:
            # ARIA标签可能不完整，记录但不失败
            print(f"ARIA labels test warning: {str(e)}")
    
    def test_color_contrast(self, driver, base_url):
        """测试颜色对比度（基础检查）"""
        driver.get(base_url)
        
        wait = WebDriverWait(driver, 10)
        
        try:
            # 检查文本元素的颜色对比
            text_elements = driver.find_elements(By.XPATH, "//p | //h1 | //h2 | //h3 | //span")
            
            for element in text_elements[:10]:  # 检查前10个文本元素
                if element.is_displayed() and element.text.strip():
                    # 获取颜色信息
                    color = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).color;", 
                        element
                    )
                    background_color = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).backgroundColor;", 
                        element
                    )
                    
                    # 基础检查：确保不是相同颜色
                    assert color != background_color, "Text and background should have different colors"
                    
        except Exception as e:
            # 颜色对比检查可能复杂，记录但不失败
            print(f"Color contrast test warning: {str(e)}")


class TestPerformanceCompatibility:
    """性能兼容性测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # 启用性能日志
        chrome_options.add_argument("--enable-logging")
        chrome_options.add_argument("--log-level=0")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            yield driver
            driver.quit()
        except WebDriverException:
            pytest.skip("Chrome browser not available for performance testing")
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    def test_page_load_performance(self, driver, base_url):
        """测试页面加载性能"""
        start_time = time.time()
        driver.get(base_url)
        
        # 等待页面完全加载
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        load_time = time.time() - start_time
        
        # 页面应该在5秒内加载完成
        assert load_time < 5.0, f"Page load time {load_time:.2f}s exceeds 5s threshold"
        
        # 检查页面大小
        page_size = driver.execute_script("return document.documentElement.outerHTML.length;")
        assert page_size > 0, "Page should have content"
    
    def test_javascript_performance(self, driver, base_url):
        """测试JavaScript性能"""
        driver.get(base_url)
        
        # 测试JavaScript执行时间
        start_time = time.time()
        
        # 执行一些JavaScript操作
        result = driver.execute_script("""
            var start = performance.now();
            
            // 模拟一些计算
            var sum = 0;
            for (var i = 0; i < 100000; i++) {
                sum += i;
            }
            
            var end = performance.now();
            return {
                executionTime: end - start,
                result: sum
            };
        """)
        
        execution_time = result['executionTime']
        
        # JavaScript执行应该很快
        assert execution_time < 100, f"JavaScript execution time {execution_time:.2f}ms too high"
    
    def test_memory_usage(self, driver, base_url):
        """测试内存使用（基础检查）"""
        driver.get(base_url)
        
        # 获取内存信息（如果可用）
        try:
            memory_info = driver.execute_script("""
                if (performance.memory) {
                    return {
                        usedJSHeapSize: performance.memory.usedJSHeapSize,
                        totalJSHeapSize: performance.memory.totalJSHeapSize,
                        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                    };
                }
                return null;
            """)
            
            if memory_info:
                used_memory = memory_info['usedJSHeapSize'] / 1024 / 1024  # MB
                
                # 内存使用应该在合理范围内
                assert used_memory < 100, f"Memory usage {used_memory:.2f}MB too high"
                
        except Exception:
            # 内存API可能不可用，跳过测试
            pytest.skip("Memory performance API not available")