# Open WebUI Security Pipelines

A collection of security-focused functions and pipelines for OpenWebUI, designed to enhance security team operations.

## Status Overview

| Function            | Status              | Description                              |
| ------------------- | ------------------- | ---------------------------------------- |
| TokenGuard          | Proof of Concept  | Early development, experimental use only |
| Anthropic Pipeline  | Production Ready* | Full implementation available            |
| Cloudflare WorkerAI | Beta (v0.x)       | Early release, under active development  |

*Review and run at your own risk

## Available Functions

### [TokenGuard](https://github.com/christian-taillon/open-webui-pipelines/blob/main/function/tokenGuard.py)
**Status: Proof of Concept 0.0.2**
- Early development phase
- Experimental implementation for token management
- Not recommended for production use

### Manifold Pipelines
Located in the `manifoldPipe/` directory:

#### [Anthropic Pipeline](https://github.com/christian-taillon/open-webui-pipelines/blob/main/manifoldPipe/anthropic.py)
**Status: Production Ready 0.3.17**
- Full implementation for security analysis
- Integrated with Anthropic's AI models
- Production-ready but requires thorough testing in your environment

#### [Cloudflare WorkerAI Pipeline](https://github.com/christian-taillon/open-webui-pipelines/blob/main/manifoldPipe/cloudflare.py)
**Status: Beta v0.1.6**
- Early release version
- Active development and improvements ongoing
- Basic functionality available

## Installation Instructions

1. Access your OpenWebUI instance
2. Navigate to: Workspaces -> Functions -> New Function (+ Icon)
3. Choose the function you want to implement:
   - Copy the content of the desired `.py` file
   - Paste it into the new function window
   - Save the function

### CloudFlare WorkerAI Integration
For CloudFlare WorkerAI, you can also obtain this from OpenWebUI's extension community site: [CloudFlare WorkerAI](https://openwebui.com/f/christiant/cloudfare_workerai/)

## Function Selection Guide

Choose functions based on your security team's needs and stability requirements:

- **TokenGuard**: 
  - Experimental use only
  - Suitable for testing and development environments

- **Anthropic Pipeline**: 
  - Ready for production implementation
  - Requires review and proper testing in your environment

- **Cloudflare WorkerAI Pipeline**: 
  - Beta implementation
  - Suitable for non-critical deployments
  - Expect updates and changes

## Contributing

Feel free to contribute to this project by submitting pull requests or creating issues for bugs and feature requests. I will update the content on [Open WebUI](https://openwebui.com/) manually. 

## Author

Created and maintained by Christian Taillon

## Support

For support or questions, please open an issue in the repository or contact through OpenWebUI's community channels.

---

**⚠️ Important Notice**: These functions are specifically designed for security team operations. Please review the code and ensure it meets your organization's security requirements before implementation. Pay attention to the status of each function when deciding on implementation.