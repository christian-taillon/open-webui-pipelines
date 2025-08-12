# Open WebUI Security Pipelines

A collection of security-focused functions and pipelines for OpenWebUI, designed to enhance security team operations.

## Status Overview

| Function                   | Status              | Description                              |
| -------------------------- | ------------------- | ---------------------------------------- |
| TokenGuard                 | Proof of Concept    | Early development, experimental use only |
| LlamaGuard                 | Proof of Concept    | Content filtering and safety analysis    |
| Anthropic Pipeline         | Production Ready*   | Full implementation available            |
| Anthropic Pipeline Sonnet3.7 | Beta (v0.2.x)     | Enhanced capabilities, under testing     |
| Cloudflare WorkerAI        | Beta (v0.x)         | Early release, under active development  |
| ThreatConnect IOC Enrichment | Beta (v0.7) | Indicator enrichment using ThreatConnect |

*Review and run at your own risk

## Manifold Pipelines
Located in the `manifoldPipe/` directory:

### [Anthropic Manifold Pipe](https://openwebui.com/f/justinrahb/anthropic)
**Status: Production Ready 0.3.17**
- Full implementation for security analysis
- Integrated with Anthropic's AI models
- Production-ready but requires thorough testing in your environment
- [BETA: Anthropic Manifold Pipe Beta](https://openwebui.com/f/christiant/anthropic_manifold_pipe_sonnet_3_7)

### [Cloudflare WorkerAI Pipeline](https://openwebui.com/f/christiant/cloudfare_workerai)
**Status: Beta v0.1.6**
- Early release version
- Active development and improvements ongoing
- Basic functionality available

## Functions
Located in the `function/` directory:
All function code is available in this repository. The links point to the corresponding plugins in the OpenWebUI community. While these functions are developed for compatibility with OpenWebUI pipelines, they can be adapted to other frameworks with minimal modifications.

### [TokenGuard](https://openwebui.com/f/christiant/tokenguard)
**Status: Proof of Concept 0.0.2**
- Early development phase
- Experimental implementation for token management
- Not recommended for production use

### [LlamaGuard Filter](https://openwebui.com/f/christiant/llamagaurd)
**Status: Production Ready 0.0.2**
- Content filtering using Llama Guard model
- Configurable safety categories (S1-S13)
- Provides detailed safety violation explanations

### [ThreatConnect IOC Enrichment](https://openwebui.com/f/christiant/threatconnect_ioc_enrichment)
**Status: Production Ready v0.7**
- Automatically extracts and enriches indicators of compromise
- Uses ThreatConnect's threat intelligence platform
- Provides detailed threat intelligence enrichment for indicators found in user messages
- Supports both token and access key authentication methods

## Tools
Located in the `tools/` directory:

### [Enhanced Web Scrape](https://openwebui.com/t/christiant/endgame_web_scrape)
**Status: Beta v0.1.2**
- Improved web scraping tool using Jina Reader
- Features better filtering, user configuration, and UI feedback
- Supports retries and batch scraping capabilities
- Clean content option to reduce token usage

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

- **LlamaGuard Filter**:
  - Production-ready content filtering with configurable safety categories
  - Requires Ollama installation and LlamaGuard model
  - Suitable for automated content moderation and safety compliance

- **ThreatConnect IOC Enrichment**:
  - Production-ready threat intelligence integration
  - Requires ThreatConnect API access (token or access key)
  - Automatically identifies and enriches indicators of compromise in user messages

- **Anthropic Pipeline**: 
  - Ready for production implementation
  - Requires review and proper testing in your environment

- **Cloudflare WorkerAI Pipeline**: 
  - Beta implementation
  - Suitable for non-critical deployments
  - Expect updates and changes

- **Enhanced Web Scrape Tool**:
  - Beta implementation for improved web content extraction
  - Useful for security research and threat intelligence gathering
  - Configurable cleaning options to reduce token usage

## Contributing

Feel free to contribute to this project by submitting pull requests or creating issues for bugs and feature requests. I will update the content on [Open WebUI](https://openwebui.com/) manually. 

## Author

Created and maintained by Christian Taillon

## Support

For support or questions, please open an issue in the repository or contact through OpenWebUI's community channels.

---

## Security Disclaimer

The security functions provided in this repository come with varying levels of maturity and testing:

- **Proof of Concept (PoC)** functions are experimental and may contain bugs or security flaws. They should never be relied upon as your sole security mechanism.
- **Beta** functions have undergone limited testing but may still contain unknown issues.
- Even **Production Ready** functions should be thoroughly tested in your specific environment before deployment.

These tools are meant to enhance your security posture but should be implemented as part of a defense-in-depth strategy. The author and contributors are not responsible for security incidents resulting from the implementation of these functions.

Users should conduct their own security reviews of this code before implementation in sensitive environments.