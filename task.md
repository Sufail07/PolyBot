> **Objective:** Adjust the Polymarket market scanner to create a usable Phase 1 dataset for social-signal correlation testing.
>
> **Tasks:**
>
> 1. Fetch **active Polymarket markets** using the API. Use a large `limit` (e.g., 200–500) and paginate with `offset` if needed.
> 2. Log  **all fetched markets** , including:
>    * Market ID, question, category
>    * Outcomes and their current probabilities (`price`)
>    * Volume in USD
>    * Resolution date
>    * Bid-ask spread (if available)
> 3. Modify eligibility rules:
>    * **Do not discard markets outright** if all outcomes are outside 30–70%
>    * Instead, **mark which markets fall in the 30–70% range** for reference
>    * Optionally, calculate **probability delta over 24–48h** to detect markets with high momentum
> 4. Store all markets in the database, including metadata for later filtering.
> 5. Output a summary report:
>    * Total markets fetched
>    * Number of markets with at least one outcome in 30–70% range
>    * Probability distribution statistics
>    * List of top candidates based on volume, resolution date, and momentum
> 6. Prepare the dataset for **Phase 2** where social media signals (X, Reddit) will be correlated with market movement.

> **Constraints:**
>
> * Avoid filtering too aggressively at this stage; the goal is  **data coverage** , not strict eligibility
> * Ensure all date parsing handles ISO 8601 properly
> * Ensure database updates don’t fail on duplicates

> **Output:** A fully populated database of recent markets, annotated with candidate eligibility and probability metadata, ready for social signal testing.
>
