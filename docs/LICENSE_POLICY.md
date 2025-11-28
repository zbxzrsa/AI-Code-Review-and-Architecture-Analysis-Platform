# License Policy Configuration

## Allowed Licenses

### Permissive Licenses (✅ Automatically Allowed)

- **MIT** - Massachusetts Institute of Technology License
- **Apache-2.0** - Apache License 2.0
- **BSD-2-Clause** - BSD 2-Clause "Simplified" License
- **BSD-3-Clause** - BSD 3-Clause "New" or "Revised" License
- **ISC** - ISC License
- **Unlicense** - The Unlicense
- **CC0-1.0** - Creative Commons Zero v1.0 Universal

### Weak Permissive (⚠️ Requires Review)

- **BSD-4-Clause** - BSD 4-Clause "Original" License
- **MIT-0** - MIT No Attribution
- **CC-BY-4.0** - Creative Commons Attribution 4.0
- **WTFPL** - Do What The F\*ck You Want To Public License

### Copyleft Licenses (🟡 Requires Legal Review)

- **LGPL-2.0**, **LGPL-2.1**, **LGPL-3.0** - GNU Lesser General Public License
- **MPL-2.0** - Mozilla Public License 2.0
- **EPL-2.0** - Eclipse Public License 2.0
- **CPL-1.0** - Common Public License 1.0

### Restricted Licenses (❌ Not Allowed)

- **GPL-2.0**, **GPL-3.0** - GNU General Public License
- **AGPL-3.0** - GNU Affero General Public License 3.0
- **SSPL** - Server Side Public License
- **EUPL-1.2** - European Union Public License 1.2
- **Proprietary** - Commercial/Proprietary licenses

## Special Cases

### Dual-Licensed Packages

- Packages with dual licensing options are allowed if at least one option is from the "Allowed" list
- Example: MPL-2.0 OR GPL-2.0 (allowed due to MPL option)

### Runtime Dependencies

- Runtime dependencies with restrictive licenses may be allowed if:
  1. They are not distributed with the application
  2. They are clearly documented as external dependencies
  3. Legal team has approved the usage

### Development Tools

- Build tools, test frameworks, and development dependencies may have more permissive policies
- However, they must not be distributed in the final product

## Compliance Rules

### Backend (Python)

1. **Direct Dependencies**: Must be from "Allowed" or "Weak Permissive" lists
2. **Transitive Dependencies**: Must not introduce "Restricted" licenses
3. **AI/ML Libraries**: Special attention to model licenses and data usage terms

### Frontend (JavaScript/TypeScript)

1. **NPM Packages**: Must be from "Allowed" list for production dependencies
2. **Development Dependencies**: "Weak Permissive" allowed with documentation
3. **Bundled Code**: All bundled code must comply with "Allowed" licenses

### Container Images

1. **Base Images**: Must use officially supported base images
2. **System Packages**: Must comply with "Allowed" licenses
3. **Security Scanning**: All layers must be scanned for license compliance

## Enforcement

### Automated Checks

- **CI/CD Pipeline**: Automatic license scanning on all PRs
- **Pre-commit Hooks**: Local license validation
- **SBOM Analysis**: License checking in Software Bill of Materials

### Manual Review Process

1. **Exception Request**: Document business justification
2. **Legal Review**: Legal team assessment
3. **Security Review**: Security team impact assessment
4. **Approval**: Written approval from both teams
5. **Documentation**: Clear documentation of exception

### Monitoring

- **Continuous Scanning**: Daily automated license checks
- **Alerting**: Immediate alerts for license violations
- **Reporting**: Monthly license compliance reports

## Risk Assessment

### High Risk (🔴)

- AGPL-3.0 licensed packages in networked applications
- GPL-3.0 licensed libraries in commercial products
- SSPL licensed components

### Medium Risk (🟡)

- LGPL licensed components with dynamic linking
- MPL licensed code with modifications
- Copyleft licenses in core functionality

### Low Risk (🟢)

- Permissive licensed dependencies
- Development tools with restrictive licenses
- Well-documented dual-licensed packages

## Documentation Requirements

### README.md

- License section with clear policy statement
- List of all licensed components
- Exception documentation (if any)

### Package Documentation

- License file (LICENSE) in each package
- Third-party license acknowledgments
- License compliance statements

### SBOM Documentation

- Complete license information in SBOM
- License analysis reports
- Exception justifications

## Tools and Integration

### Scanning Tools

- **FOSSA**: For comprehensive license analysis
- **pip-licenses**: Python package license checking
- **license-checker**: Node.js license validation
- **CycloneDX**: SBOM generation with license data

### CI/CD Integration

- **GitHub Actions**: Automated license scanning
- **Pre-commit Hooks**: Local license validation
- **Dependency Track**: SBOM submission and tracking

### Reporting

- **License Dashboard**: Real-time license compliance status
- **Alert System**: Immediate notification of violations
- **Compliance Reports**: Regular compliance status reports

## Contact and Escalation

### Primary Contacts

- **Legal Team**: legal@company.com
- **Security Team**: security@company.com
- **Engineering Lead**: eng-lead@company.com

### Escalation Process

1. **Detection**: Automated tool detects license violation
2. **Alert**: Immediate notification to relevant teams
3. **Assessment**: 24-hour assessment period
4. **Resolution**: Fix or exception approval
5. **Documentation**: Update documentation and processes

## Review Schedule

### Monthly Reviews

- License policy review
- Exception assessment
- Tool effectiveness evaluation
- Compliance status update

### Quarterly Reviews

- Legal team consultation
- Policy updates
- Risk assessment updates
- Training and awareness programs

### Annual Reviews

- Complete policy overhaul
- Industry best practice review
- Tool evaluation and updates
- Compliance audit results

---

**Policy Version**: 1.0  
**Last Updated**: 2025-11-27  
**Next Review**: 2025-12-27  
**Approved By**: Legal & Security Teams
