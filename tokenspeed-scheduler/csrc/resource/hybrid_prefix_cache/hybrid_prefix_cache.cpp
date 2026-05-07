// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/mamba_chunk_allocator.h"

namespace tokenspeed {

HybridPrefixCache::HybridPrefixCache(KVPrefixCache& kv_prefix_cache, MambaChunkAllocator* mamba_allocator)
    : kv_prefix_cache_{kv_prefix_cache}, mamba_allocator_{mamba_allocator}, mamba_eviction_manager_{mamba_allocator} {}

MatchResult HybridPrefixCache::Match(const token_vec_t& token_ids, std::int32_t lora_id) {
    auto match = kv_prefix_cache_.Match(token_ids, lora_id);
    augmentMatch(match);
    return match;
}

MatchResult HybridPrefixCache::Match(const std::vector<std::span<const std::int32_t>>& token_pages,
                                     std::int32_t lora_id) {
    auto match = kv_prefix_cache_.Match(token_pages, lora_id);
    augmentMatch(match);
    return match;
}

void HybridPrefixCache::augmentMatch(MatchResult& match) const {
    TreeNode* kv_terminal = match.device.last_node;
    if (kv_terminal == nullptr || kv_terminal->IsRoot()) return;

    TreeNode* mamba_node = FindLastMambaNode(kv_terminal);
    if (mamba_node == nullptr) {
        TreeNode* root = kv_terminal;
        while (!root->IsRoot()) root = root->Parent();
        match.device.last_node = root;
        match.host.last_node = root;
        return;
    }

    std::int32_t page_size = match.device.page_size;
    std::int32_t kv_depth = match.device.DepthInPage();
    std::int32_t mamba_depth = mamba_node->DepthInPage(page_size);

    match.mamba_cow_src_index = mamba_node->MambaSlotIndex();

    if (kv_depth > mamba_depth) {
        match.mamba_branching_seqlen = kv_depth * page_size;
    }

    match.device.last_node = mamba_node;
    match.host.last_node = mamba_node;
}

TreeNode* HybridPrefixCache::FindLastMambaNode(TreeNode* from) const {
    for (TreeNode* cur = from; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        if (cur->HasMamba()) return cur;
    }
    return nullptr;
}

bool HybridPrefixCache::EnsureMambaCapacityByEvict(std::int32_t num_slots) {
    return mamba_eviction_manager_.EnsureCapacity(num_slots);
}

void HybridPrefixCache::InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
    if (terminal_node == nullptr || slot == nullptr) return;
    terminal_node->AttachMamba(std::move(slot));
    mamba_eviction_manager_.TrackNode(terminal_node);
}

void HybridPrefixCache::OnKVEvict(TreeNode* node) {
    if (node == nullptr || !node->HasMamba()) return;
    mamba_eviction_manager_.UntrackNode(node);
    node->DetachMamba();
    if (node->Parent() != nullptr) {
        mamba_eviction_manager_.UpdateLeaf(node->Parent());
    }
}

std::int32_t HybridPrefixCache::AvailableSlots() const {
    return mamba_allocator_->AvailableSlots();
}

}  // namespace tokenspeed
