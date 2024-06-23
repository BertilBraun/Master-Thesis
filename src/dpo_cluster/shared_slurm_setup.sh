
# Function to check if extension to the workspace is needed
check_and_extend_workspace() {
    workspace=$1
    days=$2

    # Get the remaining time for the workspace
    remaining_time=$(ws_list | grep -A 1 "id: $workspace" | grep 'remaining time' | awk '{print $4}')

    # Check if remaining time is less than specified days
    if [ "$remaining_time" != "" ] && [ "$remaining_time" -lt "$days" ]; then
        echo "Extending workspace $workspace for 60 days."
        ws_extend "$workspace" 60
    else
        echo "No extension needed for workspace $workspace."
    fi
}



# Check and extend workspaces
check_and_extend_workspace MA 15

source ~/.bashrc
# Export paths that are shared between different scripts