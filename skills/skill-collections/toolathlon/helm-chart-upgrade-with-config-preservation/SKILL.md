---
name: helm-chart-upgrade-with-config-preservation
description: Upgrades a Helm chart to a specific version while preserving existing configurations from a values file and ensuring the application remains operational. Handles the complete upgrade workflow including state discovery, compatibility resolution, configuration updates, execution, monitoring, and verification.
---
# Instructions

## 1. Discover Current State & Environment
- **Locate Workspace Files:** List the workspace directory to find configuration files (`config/`) and kubeconfigs (`k8s_configs/`).
- **Check Kubernetes Context:** Verify the active or available Kubernetes cluster contexts.
- **Read Values File:** Load the user-specified values YAML file (e.g., `config/redis-values.yaml`) to understand the current configuration.
- **Verify Target Namespace:** Confirm the target namespace (e.g., `shared-services`) exists.
- **Check Current Helm Release:** Use `helm list -n <namespace>` with the correct kubeconfig to find the current chart name and version.
- **Check Pod Status:** List pods in the target namespace to establish a baseline health state.

## 2. Identify & Resolve Compatibility Issues
- **Compare Values:** Retrieve the previously applied Helm values (`helm get values --revision <N>`) to identify any runtime configurations (like image repositories) not present in the static values file.
- **Analyze Upgrade Errors:** If the initial upgrade fails (e.g., due to image pull errors or security warnings), inspect the pod events and StatefulSet specs to diagnose the root cause (e.g., `ErrImagePull`).
- **Common Fixes:** Be prepared to handle:
    - **Image Repository Changes:** Newer Bitnami charts may require using `bitnamilegacy/` repositories. Add `image.repository`, `sentinel.image.repository`, `metrics.image.repository`, `volumePermissions.image.repository`, and `sysctl.image.repository` fields to the values file.
    - **Security Policy:** Add `global.security.allowInsecureImages: true` to bypass non-standard container warnings.
- **Update Values File:** Edit the user's values YAML file in-place to incorporate the necessary compatibility fixes, ensuring no duplicate sections are created.

## 3. Execute the Helm Upgrade
- **Run Upgrade Command:** Execute `helm upgrade <RELEASE_NAME> <REPO/CHART> --version <TARGET_VERSION> -n <NAMESPACE> -f <VALUES_FILE_PATH> --kubeconfig <KUBECONFIG_PATH>`.
- **Handle Kubeconfig:** If the `export` command is not allowed, pass the kubeconfig path directly via the `--kubeconfig` flag.

## 4. Monitor & Remediate Deployment
- **Watch Pod Status:** After the upgrade, immediately check the status of all pods in the namespace. Look for `Pending`, `ErrImagePull`, `ImagePullBackOff`, or `CrashLoopBackOff` states.
- **Inspect Stuck Pods:** Use `kubectl describe pod <name>` to get detailed events and identify why pods are not becoming `Ready`.
- **Force Pod Recreation:** If StatefulSets have updated revisions but old pods persist with outdated images, **delete the problematic pods** (`kubectl delete pod <name>`). The StatefulSet controller will recreate them with the new spec.
- **Wait for Readiness:** Poll pod status until all pods show `READY` (e.g., `1/1`) and `STATUS: Running`. Allow time for startup and readiness probes to pass.

## 5. Verify Successful Upgrade
- **Confirm Helm Revision:** Verify `helm list` shows the correct new chart version and `STATUS: deployed`.
- **Check StatefulSet Health:** Confirm all StatefulSets show `READY` replicas matching the desired count (e.g., `1/1`, `2/2`).
- **Validate Applied Values:** Run `helm get values` to ensure all configurations from the updated values file are present.
- **Final Health Check:** Perform a final check that all application pods are `Ready` and services exist.

## Key Triggers & User Intent
- The user requests to upgrade a specific Helm chart to an exact version.
- The user emphasizes preserving configurations from a named values YAML file.
- The user requires the application (pods) to be running properly after the upgrade.
- Keywords: `upgrade`, `helm chart`, `version X.X.X`, `values.yaml`, `ensure configurations remain effective`.

## Critical Notes
- **Image Compatibility:** The trajectory shows a common issue where Bitnami's public image registry changed. Always check and specify `bitnamilegacy/` repositories if standard pulls fail.
- **StatefulSet Rollout:** Helm upgrades may not immediately replace existing Pods in StatefulSets. Manual pod deletion is often required to force recreation with new images.
- **Security Warnings:** Adding `global.security.allowInsecureImages: true` is necessary when using non-standard images but carries security implications. The skill should document this trade-off.
- **Idempotency:** The steps are designed to be re-runnable. Checking state before making changes prevents unnecessary operations.
