## **Candidate Selection Logic — “Momentum + Liquidity Focus”**

### **1️⃣ Active Markets Only**

* Filter for  **markets that are still open** , i.e., `end_date > now`.
* Ignore closed or expired markets.

### **2️⃣ Minimum Volume Threshold**

* Keep markets **with meaningful trading activity** to avoid illiquid noise.

  Example:

  <pre class="overflow-visible! px-0!" data-start="602" data-end="670"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>MIN_VOLUME_USD = </span><span>50_000</span><span></span><span># adjust based on dataset</span><span>
  </span></span></code></div></div></pre>

### **3️⃣ Momentum-Based Filtering**

* Instead of static probability:
  * Track **probability delta** over a fixed period (e.g., 24h or 48h).
  * Markets with **high probability change (up or down)** indicate emerging trends.
  * Example:
    <pre class="overflow-visible! px-0!" data-start="919" data-end="1044"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>momentum = </span><span>abs</span><span>(current_probability - probability_24h_ago)
    MIN_MOMENTUM = </span><span>0.05</span><span></span><span># 5% move within 24h</span><span>
    </span></span></code></div></div></pre>

### **4️⃣ Spread / Liquidity Check (Optional)**

* Markets with extremely tight spreads or low liquidity are harder to predict reliably.
* If API provides `bid_ask_spread`, filter out markets with very wide spreads:
  <pre class="overflow-visible! px-0!" data-start="1265" data-end="1307"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>MAX_SPREAD = </span><span>0.10</span><span></span><span># 10%</span><span>
  </span></span></code></div></div></pre>

### **5️⃣ Probability Flexibility**

* Don’t restrict to 30–70%. Instead:
  * Use  **all outcome probabilities** , but **weight candidates** by:
    * Closer to 50% = higher potential movement
    * Delta over time = higher “momentum score”

### **6️⃣ Category / Topic Selection**

* If you’re testing niches (e.g., crypto, tech, politics):
  * Limit to categories of interest to  **reduce noise** .
  * Example:
    <pre class="overflow-visible! px-0!" data-start="1722" data-end="1783"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>ALLOWED_CATEGORIES = [</span><span>"Tech"</span><span>, </span><span>"Crypto"</span><span>]
    </span></span></code></div></div></pre>

### **7️⃣ Candidate Scoring**

* Assign a **score per market** to rank priorities:

  <pre class="overflow-visible! px-0!" data-start="1869" data-end="1985"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>Score = alpha * normalized_momentum + beta * normalized_volume + gamma * probability_proximity_to_50
  </span></span></code></div></div></pre>

  * α, β, γ are tunable weights (e.g., α=0.5, β=0.3, γ=0.2)
  * `probability_proximity_to_50 = 1 - abs(0.5 - current_probability) * 2` (ranges 0–1)
* Sort markets by score descending → top N candidates.

---

### **8️⃣ Candidate Output**

* For each candidate, store:
  * Market ID & Question
  * Category
  * Current outcomes & probabilities
  * 24h probability delta
  * Volume
  * Score
* These will feed Phase 2, where you track **social signals** and correlate with probability momentum.
