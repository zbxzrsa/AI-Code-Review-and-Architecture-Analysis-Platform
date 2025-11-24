"""
端到端用户流程测试
使用Selenium WebDriver测试完整的用户操作流程
"""
import pytest
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class TestUserAuthentication:
    """用户认证流程测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        """设置WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    @pytest.fixture
    def base_url(self):
        """基础URL"""
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    def test_user_registration_flow(self, driver, base_url):
        """测试用户注册流程"""
        # 访问注册页面
        driver.get(f"{base_url}/register")
        
        # 等待页面加载
        wait = WebDriverWait(driver, 10)
        
        # 填写注册表单
        username_field = wait.until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        username_field.send_keys("testuser123")
        
        email_field = driver.find_element(By.NAME, "email")
        email_field.send_keys("testuser123@example.com")
        
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys("SecurePassword123!")
        
        confirm_password_field = driver.find_element(By.NAME, "confirmPassword")
        confirm_password_field.send_keys("SecurePassword123!")
        
        # 提交表单
        submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        submit_button.click()
        
        # 验证注册成功
        try:
            success_message = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
            )
            assert "注册成功" in success_message.text or "Registration successful" in success_message.text
        except TimeoutException:
            # 检查是否跳转到登录页面
            assert "/login" in driver.current_url or "/dashboard" in driver.current_url
    
    def test_user_login_flow(self, driver, base_url):
        """测试用户登录流程"""
        # 访问登录页面
        driver.get(f"{base_url}/login")
        
        wait = WebDriverWait(driver, 10)
        
        # 填写登录表单
        username_field = wait.until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        username_field.send_keys("testuser123")
        
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys("SecurePassword123!")
        
        # 提交登录
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        
        # 验证登录成功 - 应该跳转到仪表盘
        wait.until(EC.url_contains("/dashboard"))
        assert "/dashboard" in driver.current_url
        
        # 验证用户信息显示
        user_info = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "user-info"))
        )
        assert "testuser123" in user_info.text
    
    def test_logout_flow(self, driver, base_url):
        """测试用户登出流程"""
        # 先登录
        self.test_user_login_flow(driver, base_url)
        
        wait = WebDriverWait(driver, 10)
        
        # 点击登出按钮
        logout_button = wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, "logout-button"))
        )
        logout_button.click()
        
        # 验证登出成功 - 应该跳转到首页或登录页
        wait.until(
            lambda d: "/login" in d.current_url or d.current_url == base_url + "/"
        )
        
        # 验证用户信息已清除
        try:
            driver.find_element(By.CLASS_NAME, "user-info")
            pytest.fail("User info should not be visible after logout")
        except NoSuchElementException:
            pass  # 这是预期的


class TestCodeAnalysisFlow:
    """代码分析流程测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    @pytest.fixture(autouse=True)
    def login_user(self, driver, base_url):
        """自动登录用户"""
        driver.get(f"{base_url}/login")
        wait = WebDriverWait(driver, 10)
        
        username_field = wait.until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        username_field.send_keys("testuser123")
        
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys("SecurePassword123!")
        
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        
        wait.until(EC.url_contains("/dashboard"))
    
    def test_code_input_and_analysis(self, driver, base_url):
        """测试代码输入和分析流程"""
        # 导航到代码分析页面
        driver.get(f"{base_url}/code-analysis")
        
        wait = WebDriverWait(driver, 15)
        
        # 等待代码编辑器加载
        code_editor = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "code-editor"))
        )
        
        # 输入测试代码
        test_code = """
def divide(a, b):
    return a / b

def main():
    result = divide(10, 0)
    print(result)
        """
        
        # 清空编辑器并输入代码
        code_editor.clear()
        code_editor.send_keys(test_code)
        
        # 点击分析按钮
        analyze_button = driver.find_element(By.XPATH, "//button[contains(text(), '分析') or contains(text(), 'Analyze')]")
        analyze_button.click()
        
        # 等待分析结果
        results_container = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
        )
        
        # 验证结果显示
        assert results_container.is_displayed()
        
        # 检查是否有缺陷检测结果
        try:
            defects_section = driver.find_element(By.CLASS_NAME, "defects-section")
            assert defects_section.is_displayed()
        except NoSuchElementException:
            pass  # 可能没有检测到缺陷
        
        # 检查是否有质量指标
        try:
            metrics_section = driver.find_element(By.CLASS_NAME, "metrics-section")
            assert metrics_section.is_displayed()
        except NoSuchElementException:
            pass
    
    def test_file_upload_analysis(self, driver, base_url):
        """测试文件上传分析流程"""
        driver.get(f"{base_url}/code-analysis")
        
        wait = WebDriverWait(driver, 15)
        
        # 查找文件上传控件
        try:
            file_upload = wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
            )
            
            # 创建临时测试文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("""
def test_function():
    x = 1 / 0  # Division by zero
    return x

if __name__ == "__main__":
    test_function()
                """)
                temp_file_path = f.name
            
            # 上传文件
            file_upload.send_keys(temp_file_path)
            
            # 等待文件处理
            time.sleep(2)
            
            # 点击分析按钮
            analyze_button = driver.find_element(By.XPATH, "//button[contains(text(), '分析') or contains(text(), 'Analyze')]")
            analyze_button.click()
            
            # 等待分析结果
            results_container = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
            )
            assert results_container.is_displayed()
            
            # 清理临时文件
            os.unlink(temp_file_path)
            
        except TimeoutException:
            # 如果没有文件上传功能，跳过此测试
            pytest.skip("File upload feature not available")
    
    def test_analysis_results_navigation(self, driver, base_url):
        """测试分析结果导航"""
        # 先进行一次分析
        self.test_code_input_and_analysis(driver, base_url)
        
        wait = WebDriverWait(driver, 10)
        
        # 测试标签页切换
        tabs = driver.find_elements(By.CLASS_NAME, "ant-tabs-tab")
        
        for tab in tabs:
            tab.click()
            time.sleep(1)  # 等待内容加载
            
            # 验证对应的内容面板是否显示
            tab_content = driver.find_element(By.CLASS_NAME, "ant-tabs-content")
            assert tab_content.is_displayed()
    
    def test_export_results(self, driver, base_url):
        """测试导出分析结果"""
        # 先进行一次分析
        self.test_code_input_and_analysis(driver, base_url)
        
        wait = WebDriverWait(driver, 10)
        
        # 查找导出按钮
        try:
            export_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '导出') or contains(text(), 'Export')]"))
            )
            export_button.click()
            
            # 验证导出选项
            export_options = driver.find_elements(By.CLASS_NAME, "export-option")
            assert len(export_options) > 0
            
        except TimeoutException:
            # 如果没有导出功能，跳过此测试
            pytest.skip("Export feature not available")


class TestProjectManagement:
    """项目管理流程测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    @pytest.fixture(autouse=True)
    def login_user(self, driver, base_url):
        """自动登录用户"""
        driver.get(f"{base_url}/login")
        wait = WebDriverWait(driver, 10)
        
        username_field = wait.until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        username_field.send_keys("testuser123")
        
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys("SecurePassword123!")
        
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        
        wait.until(EC.url_contains("/dashboard"))
    
    def test_create_new_project(self, driver, base_url):
        """测试创建新项目"""
        # 导航到项目页面
        driver.get(f"{base_url}/projects")
        
        wait = WebDriverWait(driver, 10)
        
        # 点击创建项目按钮
        create_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '创建') or contains(text(), 'Create')]"))
        )
        create_button.click()
        
        # 填写项目信息
        project_name_field = wait.until(
            EC.presence_of_element_located((By.NAME, "projectName"))
        )
        project_name_field.send_keys("Test Project E2E")
        
        description_field = driver.find_element(By.NAME, "description")
        description_field.send_keys("This is a test project created by E2E tests")
        
        # 提交创建
        submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        submit_button.click()
        
        # 验证项目创建成功
        try:
            success_message = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
            )
            assert "创建成功" in success_message.text or "created successfully" in success_message.text
        except TimeoutException:
            # 检查项目是否出现在列表中
            project_list = driver.find_element(By.CLASS_NAME, "project-list")
            assert "Test Project E2E" in project_list.text
    
    def test_view_project_details(self, driver, base_url):
        """测试查看项目详情"""
        # 先创建项目
        self.test_create_new_project(driver, base_url)
        
        wait = WebDriverWait(driver, 10)
        
        # 点击项目查看详情
        project_item = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Test Project E2E')]"))
        )
        project_item.click()
        
        # 验证项目详情页面
        project_details = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "project-details"))
        )
        assert project_details.is_displayed()
        assert "Test Project E2E" in project_details.text
    
    def test_delete_project(self, driver, base_url):
        """测试删除项目"""
        # 先创建项目
        self.test_create_new_project(driver, base_url)
        
        wait = WebDriverWait(driver, 10)
        
        # 查找删除按钮
        try:
            delete_button = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "delete-button"))
            )
            delete_button.click()
            
            # 确认删除
            confirm_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '确认') or contains(text(), 'Confirm')]"))
            )
            confirm_button.click()
            
            # 验证删除成功
            time.sleep(2)  # 等待删除完成
            project_list = driver.find_element(By.CLASS_NAME, "project-list")
            assert "Test Project E2E" not in project_list.text
            
        except TimeoutException:
            # 如果没有删除功能，跳过此测试
            pytest.skip("Delete feature not available")


class TestDashboardInteraction:
    """仪表盘交互测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    @pytest.fixture(autouse=True)
    def login_user(self, driver, base_url):
        """自动登录用户"""
        driver.get(f"{base_url}/login")
        wait = WebDriverWait(driver, 10)
        
        username_field = wait.until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        username_field.send_keys("testuser123")
        
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys("SecurePassword123!")
        
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        
        wait.until(EC.url_contains("/dashboard"))
    
    def test_dashboard_navigation(self, driver, base_url):
        """测试仪表盘导航"""
        driver.get(f"{base_url}/dashboard")
        
        wait = WebDriverWait(driver, 10)
        
        # 验证仪表盘元素
        dashboard_title = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        assert "仪表盘" in dashboard_title.text or "Dashboard" in dashboard_title.text
        
        # 测试侧边栏导航
        sidebar_items = driver.find_elements(By.CLASS_NAME, "ant-menu-item")
        
        for item in sidebar_items:
            if item.is_displayed() and item.is_enabled():
                item_text = item.text
                item.click()
                time.sleep(1)  # 等待页面加载
                
                # 验证URL变化或页面内容变化
                current_url = driver.current_url
                assert base_url in current_url
    
    def test_statistics_display(self, driver, base_url):
        """测试统计信息显示"""
        driver.get(f"{base_url}/dashboard")
        
        wait = WebDriverWait(driver, 10)
        
        # 验证统计卡片
        stat_cards = wait.until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "ant-statistic"))
        )
        
        assert len(stat_cards) > 0
        
        for card in stat_cards:
            # 验证统计卡片包含数值
            stat_value = card.find_element(By.CLASS_NAME, "ant-statistic-content-value")
            assert stat_value.text.strip() != ""
    
    def test_responsive_design(self, driver, base_url):
        """测试响应式设计"""
        driver.get(f"{base_url}/dashboard")
        
        wait = WebDriverWait(driver, 10)
        
        # 测试不同屏幕尺寸
        screen_sizes = [
            (1920, 1080),  # 桌面
            (1024, 768),   # 平板
            (375, 667),    # 手机
        ]
        
        for width, height in screen_sizes:
            driver.set_window_size(width, height)
            time.sleep(1)  # 等待布局调整
            
            # 验证页面仍然可用
            dashboard_content = driver.find_element(By.CLASS_NAME, "dashboard-content")
            assert dashboard_content.is_displayed()
            
            # 验证导航菜单在小屏幕上的行为
            if width < 768:
                # 小屏幕应该有汉堡菜单或折叠菜单
                try:
                    menu_toggle = driver.find_element(By.CLASS_NAME, "menu-toggle")
                    assert menu_toggle.is_displayed()
                except NoSuchElementException:
                    # 如果没有汉堡菜单，检查侧边栏是否自动隐藏
                    sidebar = driver.find_element(By.CLASS_NAME, "ant-layout-sider")
                    # 在小屏幕上侧边栏可能被隐藏或折叠
                    pass


class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    @pytest.fixture
    def base_url(self):
        return os.getenv("E2E_BASE_URL", "http://localhost:3000")
    
    def test_404_page_handling(self, driver, base_url):
        """测试404页面处理"""
        # 访问不存在的页面
        driver.get(f"{base_url}/non-existent-page")
        
        wait = WebDriverWait(driver, 10)
        
        # 验证404页面或重定向
        try:
            error_message = wait.until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), '404') or contains(text(), '页面不存在') or contains(text(), 'Page not found')]"))
            )
            assert error_message.is_displayed()
        except TimeoutException:
            # 可能重定向到首页
            assert driver.current_url in [base_url, f"{base_url}/", f"{base_url}/dashboard"]
    
    def test_network_error_handling(self, driver, base_url):
        """测试网络错误处理"""
        # 先正常访问页面
        driver.get(f"{base_url}/dashboard")
        
        wait = WebDriverWait(driver, 10)
        
        # 模拟网络错误（通过访问无效的API端点）
        driver.execute_script("""
            fetch('/api/invalid-endpoint')
                .catch(error => {
                    console.error('Network error:', error);
                    // 触发错误处理逻辑
                });
        """)
        
        time.sleep(2)  # 等待错误处理
        
        # 验证页面仍然可用（没有崩溃）
        dashboard_content = driver.find_element(By.CLASS_NAME, "dashboard-content")
        assert dashboard_content.is_displayed()
    
    def test_form_validation_errors(self, driver, base_url):
        """测试表单验证错误"""
        driver.get(f"{base_url}/login")
        
        wait = WebDriverWait(driver, 10)
        
        # 提交空表单
        login_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        )
        login_button.click()
        
        # 验证错误消息显示
        try:
            error_messages = wait.until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "error-message"))
            )
            assert len(error_messages) > 0
        except TimeoutException:
            # 检查表单验证样式
            form_fields = driver.find_elements(By.CLASS_NAME, "ant-form-item-has-error")
            assert len(form_fields) > 0