import { getStatementMatrix } from '$lib/db';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const rows = await getStatementMatrix();
	return { rows };
};
