# Open WebUI Security Pipelines

A collection of security-focused functions and pipelines for OpenWebUI, designed to enhance security team operations.

## Overview

This repository contains custom functions and pipelines that can be integrated into OpenWebUI instances, specifically tailored for security teams. These tools help automate and streamline security operations through AI-powered workflows.

## Available Functions

### TokenGuard (`function/tokenGuard.py`)
A security function that helps manage and protect API tokens and sensitive credentials within OpenWebUI.

### Manifold Pipelines
Located in the `manifoldPipe/` directory:

- **Anthropic Pipeline** (`anthropic.py`): Integration with Anthropic's AI models for security analysis
- **Cloudflare WorkerAI Pipeline** (`cloudflare.py`): Leverage Cloudflare's WorkerAI capabilities for security operations

## Installation Instructions

1. Access your OpenWebUI instance
2. Navigate to: Workspaces -> Functions -> New Function (+ Icon)
3. Choose the function you want to implement:
   - Copy the content of the desired `.py` file
   - Paste it into the new function window
   - Save the function

### CloudFlare WorkerAI Integration
For CloudFlare WorkerAI specific implementations, visit:
[https://openwebui.com/f/christiant/cloudfare_workerai/](https://openwebui.com/f/christiant/cloudfare_workerai/)

## Function Selection Guide

Choose functions based on your security team's needs:

- **TokenGuard**: Implement if you need enhanced token security and credential management
- **Anthropic Pipeline**: Use for advanced security analysis and threat detection
- **Cloudflare WorkerAI Pipeline**: Implement for edge-based security processing and analysis

## Contributing

Feel free to contribute to this project by submitting pull requests or creating issues for bugs and feature requests.

## License

This project is licensed under the terms included in the LICENSE file.

## Author

Created and maintained by Christian Taillon

## Support

For support or questions, please open an issue in the repository or contact through OpenWebUI's community channels.

---

**Note**: These functions are specifically designed for security team operations. Please review the code and ensure it meets your organization's security requirements before implementation.