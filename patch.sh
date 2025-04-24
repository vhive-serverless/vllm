if patch -p2 --dry-run -d "$VLLM_PATH" < "$PATCH_FILE" > /dev/null 2>&1; then
    echo "vLLM patch is not applied. Applying the patch now..."
    # Apply the patch
    patch -p2 -d "$VLLM_PATH" < "$PATCH_FILE"
    echo "Patch applied successfully."
else
    echo "vLLM patch has already been applied. Skipping..."
fi
