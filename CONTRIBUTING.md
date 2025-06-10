# 🤝 Contributing to ComfyUI Deforum-X-Flux Nodes

Thank you for your interest in contributing! This project thrives on community contributions and we welcome all forms of participation.

## 🌟 Ways to Contribute

### 🐛 Bug Reports
- **Search existing issues** before creating new ones
- **Use the bug report template** when available
- **Include system information**: OS, Python version, GPU details
- **Provide reproduction steps** with minimal examples
- **Include error messages** and stack traces

### ✨ Feature Requests
- **Check existing feature requests** to avoid duplicates
- **Describe the use case** and expected behavior
- **Provide examples** of how the feature would be used
- **Consider implementation complexity** and maintenance burden

### 📚 Documentation
- **Fix typos and grammar** in documentation
- **Add examples** and tutorials
- **Improve installation instructions**
- **Translate documentation** to other languages
- **Create video tutorials** and guides

### 💻 Code Contributions
- **Bug fixes** and performance improvements
- **New features** and enhancements
- **Test coverage** improvements
- **Code quality** and refactoring

## 🚀 Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/A043-studios/comfyui-deforum-x-flux-nodes.git
   cd comfyui-deforum-x-flux-nodes
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   ```

4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Workflow

1. **Make your changes** following the coding standards
2. **Add tests** for new functionality
3. **Run tests** to ensure everything works:
   ```bash
   python -m pytest tests/
   ```
4. **Format code** with Black:
   ```bash
   black .
   ```
5. **Check code quality** with flake8:
   ```bash
   flake8 .
   ```
6. **Commit your changes** with descriptive messages
7. **Push to your fork** and create a pull request

## 📝 Coding Standards

### Python Style
- **Follow PEP 8** with 100-character line limit
- **Use Black** for code formatting
- **Use type hints** for function signatures
- **Write docstrings** for all public functions and classes
- **Use meaningful variable names** and comments

### Code Organization
- **Keep functions small** and focused
- **Use descriptive names** for functions and variables
- **Group related functionality** in classes
- **Minimize dependencies** between modules
- **Follow ComfyUI conventions** for node implementation

### Testing
- **Write unit tests** for all new functionality
- **Include integration tests** for workflows
- **Test edge cases** and error conditions
- **Maintain test coverage** above 80%
- **Use descriptive test names** and comments

## 🎯 Node Development Guidelines

### Creating New Nodes

1. **Follow the template structure**:
   ```python
   class YourNewNode:
       @classmethod
       def INPUT_TYPES(cls):
           return {"required": {...}, "optional": {...}}
       
       RETURN_TYPES = (...)
       RETURN_NAMES = (...)
       FUNCTION = "your_function"
       CATEGORY = "Deforum-X-Flux"
       
       def your_function(self, ...):
           # Implementation
           return (result,)
   ```

2. **Add comprehensive input validation**
3. **Include proper error handling**
4. **Write clear documentation**
5. **Add example usage**

### Mathematical Expressions
- **Use numexpr** for safe expression evaluation
- **Validate expressions** before evaluation
- **Support standard mathematical functions**
- **Provide clear error messages** for invalid expressions

### Performance Considerations
- **Optimize for memory usage** with large animations
- **Use efficient tensor operations**
- **Implement proper cleanup** of resources
- **Consider GPU memory limitations**

## 📋 Pull Request Process

### Before Submitting
- [ ] **Code follows style guidelines**
- [ ] **Tests pass** locally
- [ ] **Documentation updated** if needed
- [ ] **Changelog updated** for significant changes
- [ ] **No merge conflicts** with main branch

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots/Examples
Include relevant examples or screenshots

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process
1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different platforms if needed
4. **Documentation review** for clarity
5. **Approval** and merge by maintainers

## 🐛 Issue Guidelines

### Bug Report Template
```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- ComfyUI: [version]
- GPU: [e.g., RTX 4090, 24GB VRAM]

**Additional Context**
Screenshots, logs, or other relevant information
```

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Examples, mockups, or references
```

## 🏆 Recognition

### Contributors
All contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

### Types of Contributions
- **Code**: Bug fixes, features, optimizations
- **Documentation**: Guides, examples, translations
- **Testing**: Bug reports, test cases, validation
- **Community**: Support, discussions, tutorials

## 📞 Communication

### Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration
- **Discord**: Real-time community chat (ComfyUI server)

### Guidelines
- **Be respectful** and constructive
- **Search existing discussions** before posting
- **Provide context** and relevant information
- **Follow community guidelines**

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

If you have questions about contributing:
1. **Check the documentation** first
2. **Search existing issues** and discussions
3. **Ask in GitHub Discussions**
4. **Contact maintainers** if needed

**Thank you for contributing to ComfyUI Deforum-X-Flux Nodes!** 🎬✨

Your contributions help make AI-powered video animation accessible to everyone.
